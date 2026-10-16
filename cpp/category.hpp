// CCG category representation: parsing, interning, feature unification.
//
// Categories are interned in a CategoryStore and referred to by CatId. The
// store is append-only and guarded by a shared mutex so several parser
// threads can share it.
//
// Feature conventions (C&C / CCGbank style):
//   * A bare `S` is the feature variable `S[X]`; it unifies with any S feature
//     and the binding flows into the rule result (`S/S  S[dcl] -> S[dcl]`).
//   * A bare feature on any other atom (NP, N, ...) is a wildcard that matches
//     anything but never binds (`(S[dcl]\NP)/NP  NP[expl] -> S[dcl]\NP`).
//   * Features listed in `strip_features` (default: nb) are removed at parse
//     time so `NP[nb]/N  N -> NP` as in CCGbank.
//   * `X[conj]` is a top-level marker set by the conjunction rule.
#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ccg {

using CatId = std::int32_t;
constexpr CatId kNoCat = -1;

enum class Slash : std::uint8_t { Fwd = 0, Bwd = 1 };

struct Category {
    bool atomic = true;
    bool conj = false;           // top-level [conj] marker
    std::string atom;            // atomic: base symbol (S, NP, N, PP, conj, ",", ...)
    std::string feature;         // atomic: "" (none/wildcard), "X" (variable) or concrete
    CatId result = kNoCat;       // complex: X in X|Y
    CatId arg = kNoCat;          // complex: Y in X|Y
    Slash slash = Slash::Fwd;    // complex
    std::string text;            // canonical rendering; also the intern key
};

// Binding environment for the single feature variable X.
struct Bindings {
    std::string x;
    bool bound() const { return !x.empty(); }
};

class CategoryStore {
public:
    explicit CategoryStore(std::vector<std::string> strip_features = {"nb"},
                           std::vector<std::string> punct_atoms = {",", ".", ":", ";", "LRB", "RRB", "LQU", "RQU"});

    // Parse a CCGbank category string. Throws std::invalid_argument on error.
    CatId parse(const std::string& text);
    std::optional<CatId> try_parse(const std::string& text) noexcept;

    // Elements are never moved or erased, so the returned reference stays valid.
    const Category& get(CatId id) const;
    const std::string& str(CatId id) const { return get(id).text; }
    std::size_t size() const;

    CatId make_atomic(const std::string& atom, const std::string& feature, bool conj = false);
    CatId make_complex(CatId result, Slash slash, CatId arg, bool conj = false);
    CatId set_conj(CatId id, bool conj);

    bool is_atomic(CatId id) const { return get(id).atomic; }
    bool is_conj(CatId id) const { return get(id).conj; }
    bool is_punct(CatId id) const;
    bool is_conj_atom(CatId id) const;
    bool is_modifier(CatId id) const;      // X|X
    bool is_type_raised(CatId id) const;   // T|(T|X)
    bool is_vp(CatId id) const;            // S|NP (any S feature)
    bool is_atom(CatId id, const std::string& atom) const;

    // Category with every feature removed (bare S stays the variable). Used
    // for feature-insensitive "seen rule" keys.
    CatId strip_features(CatId id);

    // Structural unification. `ba` receives bindings for variables in `a`,
    // `bb` for variables in `b`. Returns false on mismatch or binding conflict.
    bool unify(CatId a, CatId b, Bindings& ba, Bindings& bb) const;
    bool matches(CatId a, CatId b) const;

    // Replace the variable X with the bound feature (identity when unbound).
    CatId substitute(CatId id, const Bindings& b);

private:
    CatId intern(Category&& cat);
    CatId parse_expr(const std::string& s, std::size_t& pos, bool& conj_flag);
    CatId parse_term(const std::string& s, std::size_t& pos, bool& conj_flag);
    std::string render(const Category& cat) const;
    std::string render_child(CatId id) const;

    mutable std::shared_mutex mu_;
    std::deque<Category> cats_;
    std::unordered_map<std::string, CatId> index_;
    std::unordered_set<std::string> strip_features_;
    std::unordered_set<std::string> punct_atoms_;
};

}  // namespace ccg
