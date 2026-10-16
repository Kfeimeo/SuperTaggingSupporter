#include "grammar.hpp"

#include <mutex>
#include <stdexcept>

namespace ccg {

namespace {

std::uint64_t pair_key(CatId l, CatId r) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(l)) << 32) |
           static_cast<std::uint32_t>(r);
}

}  // namespace

const char* rule_name(Rule r) {
    switch (r) {
        case Rule::Lex: return "lex";
        case Rule::FA: return "fa";
        case Rule::BA: return "ba";
        case Rule::FC: return "fc";
        case Rule::GFC: return "gfc";
        case Rule::BC: return "bc";
        case Rule::BX: return "bx";
        case Rule::GBX: return "gbx";
        case Rule::Conj: return "conj";
        case Rule::Coord: return "coord";
        case Rule::LP: return "lp";
        case Rule::RP: return "rp";
        case Rule::TR: return "tr";
        case Rule::TC: return "tc";
        case Rule::Chain: return "chain";
    }
    return "?";
}

Grammar::Grammar(CategoryStore& store, RuleConfig cfg)
    : store_(store),
      cfg_(std::move(cfg)),
      left_punct_(cfg_.left_punct.begin(), cfg_.left_punct.end()),
      right_punct_(cfg_.right_punct.begin(), cfg_.right_punct.end()) {
    for (const auto& [from, to] : cfg_.type_change_rules)
        tc_rules_.emplace_back(store_.parse(from), store_.parse(to));
    for (const auto& [from, to] : cfg_.type_raise_rules)
        tr_rules_.emplace_back(store_.parse(from), store_.parse(to));
    for (const auto& root : cfg_.root_categories) roots_.push_back(store_.parse(root));
    for (const auto& [l, r] : cfg_.seen_rules) {
        auto lid = store_.try_parse(l);
        auto rid = store_.try_parse(r);
        if (!lid || !rid) continue;  // unparseable treebank artefacts are skipped
        seen_rules_.insert(pair_key(store_.strip_features(*lid), store_.strip_features(*rid)));
    }
}

int Grammar::rule_degree(Rule r) {
    switch (r) {
        case Rule::FC: case Rule::BC: case Rule::BX: return 1;
        case Rule::GFC: case Rule::GBX: return 2;
        default: return 0;
    }
}

bool Grammar::is_root(CatId cat) const {
    if (store_.is_conj(cat)) return false;
    for (CatId root : roots_)
        if (store_.matches(root, cat)) return true;
    return false;
}

bool Grammar::seen(CatId l, CatId r) {
    return seen_rules_.count(pair_key(store_.strip_features(l), store_.strip_features(r))) > 0;
}

// ---------------------------------------------------------------------------
// Memoised access
// ---------------------------------------------------------------------------

const std::vector<RuleResult>& Grammar::binary(CatId left, CatId right) {
    const std::uint64_t key = pair_key(left, right);
    {
        std::shared_lock lock(mu_);
        auto it = binary_cache_.find(key);
        if (it != binary_cache_.end()) return it->second;
    }
    std::vector<RuleResult> results = compute_binary(left, right);
    std::unique_lock lock(mu_);
    // unordered_map node references stay valid across rehashing.
    return binary_cache_.try_emplace(key, std::move(results)).first->second;
}

const std::vector<RuleResult>& Grammar::unary(CatId cat) {
    {
        std::shared_lock lock(mu_);
        auto it = unary_cache_.find(cat);
        if (it != unary_cache_.end()) return it->second;
    }
    std::vector<RuleResult> results = compute_unary(cat);
    std::unique_lock lock(mu_);
    return unary_cache_.try_emplace(cat, std::move(results)).first->second;
}

// ---------------------------------------------------------------------------
// Binary rules
// ---------------------------------------------------------------------------

std::vector<RuleResult> Grammar::compute_binary(CatId l, CatId r) {
    std::vector<RuleResult> out;
    if (cfg_.use_seen_rules && !seen(l, r)) return out;

    const Category& L = store_.get(l);
    const Category& R = store_.get(r);
    const bool lp = store_.is_punct(l);
    const bool rp = store_.is_punct(r);
    const int cap = cfg_.max_composition_degree;

    // Punctuation absorption (NF-transparent).
    if (rp && !lp && !L.conj && right_punct_.count(R.atom)) {
        RuleResult rr;
        rr.cat = l; rr.rule = Rule::RP; rr.head = 0; rr.inherit_left_nf = true;
        out.push_back(rr);
    }
    if (lp && !rp && !R.conj && left_punct_.count(L.atom)) {
        RuleResult rr;
        rr.cat = r; rr.rule = Rule::LP; rr.head = 1; rr.inherit_right_nf = true;
        out.push_back(rr);
    }

    // Conjunction: conj X -> X[conj]   (also `, X` and `; X`)
    const bool conj_left = store_.is_conj_atom(l) ||
                           (cfg_.comma_as_conj && lp && (L.atom == "," || L.atom == ";"));
    if (conj_left && !R.conj && !rp && !store_.is_conj_atom(r)) {
        RuleResult rr;
        rr.cat = store_.set_conj(r, true); rr.rule = Rule::Conj; rr.head = 1;
        out.push_back(rr);
    }

    // Coordination: X X[conj] -> X
    if (R.conj && !L.conj && !lp && !store_.is_conj_atom(l)) {
        CatId base = store_.set_conj(r, false);
        Bindings bl, br;
        if (store_.unify(l, base, bl, br)) {
            RuleResult rr;
            rr.cat = l; rr.rule = Rule::Coord; rr.head = 0;
            out.push_back(rr);
        }
    }

    // Punctuation, conj atoms and [conj]-marked categories take part in no
    // combinatory rule.
    if (lp || rp || L.conj || R.conj || store_.is_conj_atom(l) || store_.is_conj_atom(r)) return out;

    const bool l_fwd = !L.atomic && L.slash == Slash::Fwd;
    const bool l_bwd = !L.atomic && L.slash == Slash::Bwd;
    const bool r_fwd = !R.atomic && R.slash == Slash::Fwd;
    const bool r_bwd = !R.atomic && R.slash == Slash::Bwd;

    // > : X/Y Y -> X
    if (l_fwd) {
        Bindings bl, br;
        if (store_.unify(L.arg, r, bl, br)) {
            RuleResult rr;
            rr.cat = store_.substitute(L.result, bl); rr.rule = Rule::FA; rr.head = 0;
            out.push_back(rr);
        }
    }
    // < : Y X\Y -> X
    if (r_bwd) {
        Bindings br, bl;
        if (store_.unify(R.arg, l, br, bl)) {
            RuleResult rr;
            rr.cat = store_.substitute(R.result, br); rr.rule = Rule::BA; rr.head = 1;
            out.push_back(rr);
        }
    }
    // >B : X/Y Y/Z -> X/Z
    if (cap >= 1 && l_fwd && r_fwd) {
        Bindings bl, br;
        if (store_.unify(L.arg, R.result, bl, br)) {
            RuleResult rr;
            rr.cat = store_.make_complex(store_.substitute(L.result, bl), Slash::Fwd, store_.substitute(R.arg, br));
            rr.rule = Rule::FC; rr.head = 0; rr.nf_kind = NfKind::FwdComp; rr.nf_degree = 1;
            out.push_back(rr);
        }
    }
    // >B2 : X/Y (Y/Z)/W -> (X/Z)/W
    if (cap >= 2 && l_fwd && r_fwd) {
        const Category& Rr = store_.get(R.result);
        if (!Rr.atomic && Rr.slash == Slash::Fwd) {
            Bindings bl, br;
            if (store_.unify(L.arg, Rr.result, bl, br)) {
                CatId inner = store_.make_complex(store_.substitute(L.result, bl), Slash::Fwd, store_.substitute(Rr.arg, br));
                RuleResult rr;
                rr.cat = store_.make_complex(inner, Slash::Fwd, store_.substitute(R.arg, br));
                rr.rule = Rule::GFC; rr.head = 0; rr.nf_kind = NfKind::FwdComp; rr.nf_degree = 2;
                out.push_back(rr);
            }
        }
    }
    // <B : Y\Z X\Y -> X\Z
    if (cfg_.harmonic_backward_composition && cap >= 1 && l_bwd && r_bwd) {
        Bindings br, bl;
        if (store_.unify(R.arg, L.result, br, bl)) {
            RuleResult rr;
            rr.cat = store_.make_complex(store_.substitute(R.result, br), Slash::Bwd, store_.substitute(L.arg, bl));
            rr.rule = Rule::BC; rr.head = 1; rr.nf_kind = NfKind::BwdComp; rr.nf_degree = 1;
            out.push_back(rr);
        }
    }
    // <Bx : Y/Z X\Y -> X/Z
    if (cfg_.crossed_composition && cap >= 1 && l_fwd && r_bwd &&
        (!cfg_.bx_require_vp || store_.is_vp(L.result))) {
        Bindings br, bl;
        if (store_.unify(R.arg, L.result, br, bl)) {
            RuleResult rr;
            rr.cat = store_.make_complex(store_.substitute(R.result, br), Slash::Fwd, store_.substitute(L.arg, bl));
            rr.rule = Rule::BX; rr.head = 1; rr.nf_kind = NfKind::BwdCrossed; rr.nf_degree = 1;
            out.push_back(rr);
        }
    }
    // <Bx2 : (Y/Z)/W X\Y -> (X/Z)/W
    if (cfg_.crossed_composition && cap >= 2 && l_fwd && r_bwd) {
        const Category& Lr = store_.get(L.result);
        if (!Lr.atomic && Lr.slash == Slash::Fwd && (!cfg_.bx_require_vp || store_.is_vp(Lr.result))) {
            Bindings br, bl;
            if (store_.unify(R.arg, Lr.result, br, bl)) {
                CatId inner = store_.make_complex(store_.substitute(R.result, br), Slash::Fwd, store_.substitute(Lr.arg, bl));
                RuleResult rr;
                rr.cat = store_.make_complex(inner, Slash::Fwd, store_.substitute(L.arg, bl));
                rr.rule = Rule::GBX; rr.head = 1; rr.nf_kind = NfKind::BwdCrossed; rr.nf_degree = 2;
                out.push_back(rr);
            }
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Unary rules
// ---------------------------------------------------------------------------

std::vector<RuleResult> Grammar::compute_unary(CatId c) {
    std::vector<RuleResult> out;
    if (store_.is_punct(c) || store_.is_conj(c) || store_.is_conj_atom(c)) return out;
    if (cfg_.type_changing) {
        for (const auto& [from, to] : tc_rules_) {
            if (!store_.matches(from, c)) continue;
            RuleResult rr;
            rr.cat = to; rr.rule = Rule::TC; rr.head = 0;
            out.push_back(rr);
        }
    }
    if (cfg_.type_raising) {
        for (const auto& [from, to] : tr_rules_) {
            if (!store_.matches(from, c)) continue;
            RuleResult rr;
            rr.cat = to; rr.rule = Rule::TR; rr.head = 0; rr.nf_kind = NfKind::TypeRaised;
            out.push_back(rr);
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Eisner normal form
// ---------------------------------------------------------------------------
//
// NF1: the output of >B^n (n>=1) is not the primary functor of > or >B^m whenever
//      the alternative derivation fits in the degree cap (n-1+m <= cap).
// NF2: mirror image for the output of <B^n as the primary functor of <, <B^m, <Bx^m.
// NF3: a type-raised category is never the primary functor of application.
// The output of crossed composition is deliberately unconstrained: in
// `was not scary` the <Bx result `(was not)` has no adjacent alternative and
// must act as the functor of the following application.

bool Grammar::nf_allows(const RuleResult& rr, NfKind lk, int ld, NfKind rk, int rd) const {
    const int cap = cfg_.max_composition_degree;
    const int m = rule_degree(rr.rule);
    switch (rr.rule) {
        case Rule::FA:
        case Rule::FC:
        case Rule::GFC:
            if (lk == NfKind::FwdComp && ld - 1 + m <= cap) return false;
            if (rr.rule == Rule::FA && lk == NfKind::TypeRaised) return false;
            return true;
        case Rule::BA:
        case Rule::BC:
        case Rule::BX:
        case Rule::GBX:
            if (rk == NfKind::BwdComp && rd - 1 + m <= cap) return false;
            if (rr.rule == Rule::BA && rk == NfKind::TypeRaised) return false;
            return true;
        default:
            return true;
    }
}

}  // namespace ccg
