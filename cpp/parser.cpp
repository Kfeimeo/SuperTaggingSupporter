#include "parser.hpp"

#include <algorithm>
#include <mutex>

namespace ccg {

// ---------------------------------------------------------------------------
// Lexicon
// ---------------------------------------------------------------------------

const LexCandidate& Lexicon::lookup(const std::string& text) {
    {
        std::shared_lock lock(mu_);
        auto it = cache_.find(text);
        if (it != cache_.end()) return it->second;
    }
    LexCandidate cand;
    cand.text = text;
    std::size_t pos = 0;
    while (true) {
        std::size_t arrow = text.find("-->", pos);
        cand.segments.push_back(text.substr(pos, arrow == std::string::npos ? std::string::npos : arrow - pos));
        if (arrow == std::string::npos) break;
        pos = arrow + 3;
    }
    cand.valid = true;
    for (const std::string& seg : cand.segments) {
        auto id = store_.try_parse(seg);
        if (!id) { cand.valid = false; break; }
        cand.chain.push_back(*id);
    }
    if (cand.valid) cand.cat = cand.chain.back();
    std::unique_lock lock(mu_);
    return cache_.try_emplace(text, std::move(cand)).first->second;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

Parser::Parser(CategoryStore& store, Grammar& grammar, Lexicon& lexicon)
    : store_(store), grammar_(grammar), lexicon_(lexicon) {}

std::uint64_t Parser::item_key(CatId cat, NfKind kind, int degree) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(cat)) << 8) |
           (static_cast<std::uint64_t>(static_cast<std::uint8_t>(kind)) << 4) |
           static_cast<std::uint64_t>(degree & 0xF);
}

// Scores are sums of the same lexical log-probabilities in different orders,
// so "equal" is taken up to a small epsilon; the tie key then decides.
bool Parser::better(const Item& candidate, const Item& incumbent) {
    constexpr double kEps = 1e-9;
    if (candidate.score > incumbent.score + kEps) return true;
    if (candidate.score < incumbent.score - kEps) return false;
    return candidate.tie < incumbent.tie;
}

// Keep the best item per (category, NF class). Exact ties keep the first
// arrival so the result is deterministic for a given enumeration order.
bool Parser::insert(Cell& cell, Item item) {
    const std::uint64_t key = item_key(item.cat, item.nf_kind, item.nf_degree);
    auto it = cell.index.find(key);
    if (it == cell.index.end()) {
        std::int32_t idx = static_cast<std::int32_t>(pool_.size());
        pool_.push_back(item);
        cell.items.push_back(idx);
        cell.index.emplace(key, idx);
        return true;
    }
    Item& existing = pool_[it->second];
    if (better(item, existing)) {
        existing = item;
        return true;
    }
    return false;
}

void Parser::unary_closure(Cell& cell) {
    const RuleConfig& cfg = grammar_.config();
    std::vector<std::int32_t> work(cell.items.begin(), cell.items.end());
    while (!work.empty()) {
        std::int32_t idx = work.back();
        work.pop_back();
        const Item src = pool_[idx];  // copy: pool_ may grow below
        if (src.unary_depth >= cfg.max_unary_depth) continue;
        for (const RuleResult& rr : grammar_.unary(src.cat)) {
            if (rr.rule == Rule::TR && src.nf_kind == NfKind::TypeRaised) continue;  // no TR of TR
            Item item;
            item.cat = rr.cat;
            item.score = src.score + cfg.unary_penalty + (rr.rule == Rule::TR ? cfg.type_raise_penalty : 0.0);
            item.tie = src.tie;
            item.rule = rr.rule;
            item.head = 0;
            item.nf_kind = rr.nf_kind;
            item.nf_degree = 0;
            item.unary_depth = static_cast<std::uint8_t>(src.unary_depth + 1);
            item.left = idx;
            item.start = src.start;
            item.end = src.end;
            if (insert(cell, item)) {
                const std::uint64_t key = item_key(item.cat, item.nf_kind, item.nf_degree);
                work.push_back(cell.index[key]);
            }
        }
    }
}

void Parser::finalize_cell(Cell& cell) {
    const int beam = grammar_.config().cell_beam;
    if (beam <= 0 || static_cast<int>(cell.items.size()) <= beam) return;
    std::partial_sort(cell.items.begin(), cell.items.begin() + beam, cell.items.end(),
                      [this](std::int32_t a, std::int32_t b) { return pool_[a].score > pool_[b].score; });
    cell.items.resize(beam);
    cell.index.clear();
    for (std::int32_t idx : cell.items) {
        const Item& it = pool_[idx];
        cell.index.emplace(item_key(it.cat, it.nf_kind, it.nf_degree), idx);
    }
}

ParseOutput Parser::parse(const std::vector<std::string>& tokens, const Lattice& lattice) {
    ParseOutput out;
    const int n = static_cast<int>(tokens.size());
    out.chosen.assign(n, -1);
    if (n == 0 || lattice.size() != tokens.size()) return out;

    pool_.clear();
    pool_.reserve(static_cast<std::size_t>(n) * 64);
    // cell(i, j) stored at i * (n + 1) + j
    std::vector<Cell> chart(static_cast<std::size_t>(n) * (n + 1) + 1);
    auto cell_at = [&](int i, int j) -> Cell& { return chart[static_cast<std::size_t>(i) * (n + 1) + j]; };

    // Leaves
    for (int i = 0; i < n; ++i) {
        Cell& cell = cell_at(i, i + 1);
        const auto& cands = lattice[i];
        for (std::size_t ci = 0; ci < cands.size(); ++ci) {
            const LexCandidate& lex = lexicon_.lookup(cands[ci].first);
            if (!lex.valid) continue;
            Item item;
            item.cat = lex.cat;
            item.score = cands[ci].second;
            item.rule = lex.chain.size() > 1 ? Rule::Chain : Rule::Lex;
            item.nf_kind = store_.is_type_raised(lex.cat) ? NfKind::TypeRaised : NfKind::Other;
            item.unary_depth = static_cast<std::uint8_t>(std::min<std::size_t>(lex.chain.size() - 1, 255));
            item.start = static_cast<std::int16_t>(i);
            item.end = static_cast<std::int16_t>(i + 1);
            item.cand = static_cast<std::int32_t>(ci);
            insert(cell, item);
        }
        unary_closure(cell);
        finalize_cell(cell);
    }

    // Binary spans
    for (int len = 2; len <= n; ++len) {
        for (int i = 0; i + len <= n; ++i) {
            const int j = i + len;
            Cell& cell = cell_at(i, j);
            for (int k = i + 1; k < j; ++k) {
                const Cell& lc = cell_at(i, k);
                const Cell& rc = cell_at(k, j);
                if (lc.items.empty() || rc.items.empty()) continue;
                for (std::int32_t li : lc.items) {
                    const Item L = pool_[li];
                    for (std::int32_t ri : rc.items) {
                        const Item R = pool_[ri];
                        const auto& results = grammar_.binary(L.cat, R.cat);
                        if (results.empty()) continue;
                        for (const RuleResult& rr : results) {
                            if (!grammar_.nf_allows(rr, L.nf_kind, L.nf_degree, R.nf_kind, R.nf_degree)) continue;
                            Item item;
                            item.cat = rr.cat;
                            item.score = L.score + R.score;
                            item.tie = L.tie + R.tie;
                            if (rr.rule == Rule::LP || rr.rule == Rule::RP) item.tie += n - len;
                            item.rule = rr.rule;
                            item.head = rr.head;
                            if (rr.inherit_left_nf) {
                                item.nf_kind = L.nf_kind; item.nf_degree = L.nf_degree;
                            } else if (rr.inherit_right_nf) {
                                item.nf_kind = R.nf_kind; item.nf_degree = R.nf_degree;
                            } else {
                                item.nf_kind = rr.nf_kind; item.nf_degree = rr.nf_degree;
                            }
                            item.unary_depth = 0;
                            item.left = li;
                            item.right = ri;
                            item.start = static_cast<std::int16_t>(i);
                            item.end = static_cast<std::int16_t>(j);
                            insert(cell, item);
                        }
                    }
                }
            }
            unary_closure(cell);
            finalize_cell(cell);
        }
    }

    out.chart_items = pool_.size();
    const Cell& root_cell = cell_at(0, n);
    out.root_items = root_cell.items.size();
    std::int32_t best = -1;
    for (std::int32_t idx : root_cell.items) {
        const Item& it = pool_[idx];
        out.root_cell_cats.push_back(store_.str(it.cat));
        if (!grammar_.is_root(it.cat)) continue;
        if (best < 0 || better(it, pool_[best])) best = idx;
    }
    if (best < 0) return out;

    out.ok = true;
    out.score = pool_[best].score;
    out.root_cat = store_.str(pool_[best].cat);
    out.root = emit(tokens, lattice, best, out);
    return out;
}

// Convert the backpointer tree into output nodes; returns the node index.
int Parser::emit(const std::vector<std::string>& tokens, const Lattice& lattice, std::int32_t idx, ParseOutput& out) {
    const Item& it = pool_[idx];
    if (it.rule == Rule::Lex || it.rule == Rule::Chain) {
        const LexCandidate& lex = lexicon_.lookup(lattice[it.start][it.cand].first);
        out.chosen[it.start] = it.cand;
        Node leaf;
        leaf.cat = lex.segments.front();  // original lexical label (keeps e.g. NP[nb]/N)
        leaf.rule = rule_name(Rule::Lex);
        leaf.start = it.start;
        leaf.end = it.end;
        leaf.token = it.start;
        leaf.cand = it.cand;
        leaf.word = tokens[it.start];
        leaf.supertag = lex.text;
        leaf.logp = lattice[it.start][it.cand].second;
        out.nodes.push_back(std::move(leaf));
        int node = static_cast<int>(out.nodes.size()) - 1;
        // Fixed unary steps encoded in the supertag.
        for (std::size_t s = 1; s < lex.chain.size(); ++s) {
            Node u;
            u.cat = store_.str(lex.chain[s]);
            u.rule = rule_name(Rule::Chain);
            u.start = it.start;
            u.end = it.end;
            u.children = {node};
            out.nodes.push_back(std::move(u));
            node = static_cast<int>(out.nodes.size()) - 1;
        }
        return node;
    }
    Node node;
    node.cat = store_.str(it.cat);
    node.rule = rule_name(it.rule);
    node.head = it.head;
    node.start = it.start;
    node.end = it.end;
    if (it.right < 0) {
        int child = emit(tokens, lattice, it.left, out);
        node.children = {child};
    } else {
        int l = emit(tokens, lattice, it.left, out);
        int r = emit(tokens, lattice, it.right, out);
        node.children = {l, r};
    }
    out.nodes.push_back(std::move(node));
    return static_cast<int>(out.nodes.size()) - 1;
}

}  // namespace ccg
