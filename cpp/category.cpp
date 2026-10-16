#include "category.hpp"

#include <cctype>
#include <mutex>
#include <stdexcept>

namespace ccg {

namespace {

bool is_atom_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$' ||
           c == ',' || c == '.' || c == ':' || c == ';';
}

}  // namespace

CategoryStore::CategoryStore(std::vector<std::string> strip_features,
                             std::vector<std::string> punct_atoms)
    : strip_features_(strip_features.begin(), strip_features.end()),
      punct_atoms_(punct_atoms.begin(), punct_atoms.end()) {}

const Category& CategoryStore::get(CatId id) const {
    std::shared_lock lock(mu_);
    return cats_.at(static_cast<std::size_t>(id));
}

std::size_t CategoryStore::size() const {
    std::shared_lock lock(mu_);
    return cats_.size();
}

CatId CategoryStore::intern(Category&& cat) {
    {
        std::shared_lock lock(mu_);
        auto it = index_.find(cat.text);
        if (it != index_.end()) return it->second;
    }
    std::unique_lock lock(mu_);
    auto it = index_.find(cat.text);
    if (it != index_.end()) return it->second;
    CatId id = static_cast<CatId>(cats_.size());
    index_.emplace(cat.text, id);
    cats_.push_back(std::move(cat));
    return id;
}

std::string CategoryStore::render_child(CatId id) const {
    const Category& c = get(id);
    if (c.atomic) return c.text;
    return "(" + c.text + ")";
}

std::string CategoryStore::render(const Category& cat) const {
    std::string out;
    if (cat.atomic) {
        out = cat.atom;
        if (!cat.feature.empty() && cat.feature != "X") out += "[" + cat.feature + "]";
    } else {
        out = render_child(cat.result);
        out += (cat.slash == Slash::Fwd) ? '/' : '\\';
        out += render_child(cat.arg);
    }
    if (cat.conj) out += "[conj]";
    return out;
}

CatId CategoryStore::make_atomic(const std::string& atom, const std::string& feature, bool conj) {
    Category c;
    c.atomic = true;
    c.conj = conj;
    c.atom = atom;
    c.feature = feature;
    if (strip_features_.count(c.feature)) c.feature.clear();
    if (c.atom == "S" && c.feature.empty()) c.feature = "X";
    c.text = render(c);
    return intern(std::move(c));
}

CatId CategoryStore::make_complex(CatId result, Slash slash, CatId arg, bool conj) {
    // Children never carry the conj marker.
    result = set_conj(result, false);
    arg = set_conj(arg, false);
    Category c;
    c.atomic = false;
    c.conj = conj;
    c.result = result;
    c.arg = arg;
    c.slash = slash;
    c.text = render(c);
    return intern(std::move(c));
}

CatId CategoryStore::set_conj(CatId id, bool conj) {
    const Category& c = get(id);
    if (c.conj == conj) return id;
    Category copy = c;
    copy.conj = conj;
    copy.text = render(copy);
    return intern(std::move(copy));
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

CatId CategoryStore::parse_term(const std::string& s, std::size_t& pos, bool& conj_flag) {
    if (pos >= s.size()) throw std::invalid_argument("unexpected end of category: " + s);
    if (s[pos] == '(') {
        ++pos;
        CatId inner = parse_expr(s, pos, conj_flag);
        if (pos >= s.size() || s[pos] != ')') throw std::invalid_argument("missing ')' in category: " + s);
        ++pos;
        // CCGbank occasionally writes (X)[conj]; accept it.
        if (pos + 6 <= s.size() && s.compare(pos, 6, "[conj]") == 0) {
            pos += 6;
            conj_flag = true;
        }
        return inner;
    }
    std::size_t start = pos;
    while (pos < s.size() && is_atom_char(s[pos])) ++pos;
    if (start == pos) throw std::invalid_argument("bad character '" + std::string(1, s[pos]) + "' in category: " + s);
    std::string atom = s.substr(start, pos - start);
    std::string feature;
    // `S[dcl]`, `S[dcl][conj]`, `NP[conj]`: at most one feature plus the conj marker.
    while (pos < s.size() && s[pos] == '[') {
        std::size_t close = s.find(']', pos);
        if (close == std::string::npos) throw std::invalid_argument("missing ']' in category: " + s);
        std::string bracket = s.substr(pos + 1, close - pos - 1);
        pos = close + 1;
        if (bracket == "conj") {
            conj_flag = true;
        } else if (feature.empty()) {
            feature = bracket;
        } else {
            throw std::invalid_argument("multiple features in category: " + s);
        }
    }
    return make_atomic(atom, feature);
}

CatId CategoryStore::parse_expr(const std::string& s, std::size_t& pos, bool& conj_flag) {
    CatId left = parse_term(s, pos, conj_flag);
    while (pos < s.size() && (s[pos] == '/' || s[pos] == '\\')) {
        Slash slash = (s[pos] == '/') ? Slash::Fwd : Slash::Bwd;
        ++pos;
        CatId right = parse_term(s, pos, conj_flag);
        left = make_complex(left, slash, right);
    }
    return left;
}

CatId CategoryStore::parse(const std::string& text) {
    std::string s;
    s.reserve(text.size());
    for (char c : text)
        if (!std::isspace(static_cast<unsigned char>(c))) s += c;
    if (s.empty()) throw std::invalid_argument("empty category");
    std::size_t pos = 0;
    bool conj_flag = false;
    CatId id = parse_expr(s, pos, conj_flag);
    if (pos != s.size()) throw std::invalid_argument("trailing characters in category: " + text);
    if (conj_flag) id = set_conj(id, true);
    return id;
}

std::optional<CatId> CategoryStore::try_parse(const std::string& text) noexcept {
    try {
        return parse(text);
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

// ---------------------------------------------------------------------------
// Structural queries
// ---------------------------------------------------------------------------

bool CategoryStore::is_punct(CatId id) const {
    const Category& c = get(id);
    return c.atomic && !c.conj && punct_atoms_.count(c.atom) > 0;
}

bool CategoryStore::is_conj_atom(CatId id) const {
    const Category& c = get(id);
    return c.atomic && !c.conj && c.atom == "conj";
}

bool CategoryStore::is_atom(CatId id, const std::string& atom) const {
    const Category& c = get(id);
    return c.atomic && c.atom == atom;
}

bool CategoryStore::is_modifier(CatId id) const {
    const Category& c = get(id);
    return !c.atomic && c.result == c.arg;
}

bool CategoryStore::is_type_raised(CatId id) const {
    const Category& c = get(id);
    if (c.atomic) return false;
    const Category& arg = get(c.arg);
    return !arg.atomic && arg.result == c.result;
}

bool CategoryStore::is_vp(CatId id) const {
    const Category& c = get(id);
    if (c.atomic || c.slash != Slash::Bwd) return false;
    return is_atom(c.result, "S") && is_atom(c.arg, "NP");
}

CatId CategoryStore::strip_features(CatId id) {
    const Category& c = get(id);
    if (c.atomic) return make_atomic(c.atom, "", c.conj);
    return make_complex(strip_features(c.result), c.slash, strip_features(c.arg), c.conj);
}

// ---------------------------------------------------------------------------
// Unification
// ---------------------------------------------------------------------------

bool CategoryStore::unify(CatId a, CatId b, Bindings& ba, Bindings& bb) const {
    if (a == b) return true;
    const Category& A = get(a);
    const Category& B = get(b);
    if (A.conj != B.conj || A.atomic != B.atomic) return false;
    if (A.atomic) {
        if (A.atom != B.atom) return false;
        const std::string& fa = A.feature;
        const std::string& fb = B.feature;
        if (fa == fb) return true;
        if (fa == "X") {
            if (fb.empty()) return true;
            if (ba.bound()) return ba.x == fb;
            ba.x = fb;
            return true;
        }
        if (fb == "X") {
            if (fa.empty()) return true;
            if (bb.bound()) return bb.x == fa;
            bb.x = fa;
            return true;
        }
        return fa.empty() || fb.empty();
    }
    if (A.slash != B.slash) return false;
    return unify(A.result, B.result, ba, bb) && unify(A.arg, B.arg, ba, bb);
}

bool CategoryStore::matches(CatId a, CatId b) const {
    Bindings ba, bb;
    return unify(a, b, ba, bb);
}

CatId CategoryStore::substitute(CatId id, const Bindings& b) {
    if (!b.bound()) return id;
    const Category& c = get(id);
    if (c.atomic) {
        if (c.feature != "X") return id;
        return make_atomic(c.atom, b.x, c.conj);
    }
    CatId r = substitute(c.result, b);
    CatId g = substitute(c.arg, b);
    if (r == c.result && g == c.arg) return id;
    return make_complex(r, c.slash, g, c.conj);
}

}  // namespace ccg
