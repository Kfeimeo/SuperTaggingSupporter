// CCG combinatory rules, unary rules and Eisner normal-form constraints.
//
// The Grammar is NF-agnostic at the category level: binary(l, r) returns every
// rule result licensed for the two category labels and is memoised per label
// pair. Normal-form filtering (which depends on *how* a chart item was built)
// is applied separately by nf_allows().
#pragma once

#include <cstdint>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "category.hpp"

namespace ccg {

enum class Rule : std::uint8_t {
    Lex = 0,
    FA,      // >    X/Y Y -> X
    BA,      // <    Y X\Y -> X
    FC,      // >B   X/Y Y/Z -> X/Z
    GFC,     // >B2  X/Y (Y/Z)/W -> (X/Z)/W
    BC,      // <B   Y\Z X\Y -> X\Z
    BX,      // <Bx  Y/Z X\Y -> X/Z
    GBX,     // <Bx2 (Y/Z)/W X\Y -> (X/Z)/W
    Conj,    // conj X -> X[conj]
    Coord,   // X X[conj] -> X
    LP,      // punct X -> X
    RP,      // X punct -> X
    TR,      // type raising (unary)
    TC,      // type changing (unary)
    Chain,   // lexical --> chain step (unary, fixed by the supertag)
};

const char* rule_name(Rule r);

// How an item was produced, for Eisner normal-form checks.
enum class NfKind : std::uint8_t { Other = 0, FwdComp = 1, BwdComp = 2, BwdCrossed = 3, TypeRaised = 4 };

struct RuleConfig {
    int max_composition_degree = 2;
    bool harmonic_backward_composition = true;
    bool crossed_composition = true;
    bool bx_require_vp = true;        // Y in `Y/Z X\Y` must be S\NP-shaped (C&C restriction)
    bool comma_as_conj = true;        // `, X -> X[conj]` and `; X -> X[conj]`
    bool type_raising = true;
    bool type_changing = true;
    int max_unary_depth = 2;
    bool use_seen_rules = false;
    double unary_penalty = 0.0;       // added to the score of every unary step
    double type_raise_penalty = 0.0;  // added on top for type raising
    int cell_beam = 0;                // 0 = exact Viterbi; otherwise keep best N items per cell
    std::vector<std::string> left_punct = {",", ":", ";", "LRB", "LQU"};
    std::vector<std::string> right_punct = {",", ".", ":", ";", "RRB", "RQU"};
    std::vector<std::pair<std::string, std::string>> type_change_rules;  // (from pattern, to)
    std::vector<std::pair<std::string, std::string>> type_raise_rules;   // (from pattern, to)
    std::vector<std::string> root_categories = {"S"};
    std::vector<std::pair<std::string, std::string>> seen_rules;         // (left, right) label pairs
};

struct RuleResult {
    CatId cat = kNoCat;
    Rule rule = Rule::Lex;
    std::uint8_t head = 0;      // 0 = left child is head, 1 = right child
    NfKind nf_kind = NfKind::Other;
    std::uint8_t nf_degree = 0;
    bool inherit_left_nf = false;   // punctuation removal is transparent for NF
    bool inherit_right_nf = false;
};

class Grammar {
public:
    Grammar(CategoryStore& store, RuleConfig cfg);

    const RuleConfig& config() const { return cfg_; }
    CategoryStore& store() { return store_; }

    // All rule results for the label pair (memoised).
    const std::vector<RuleResult>& binary(CatId left, CatId right);
    // One-step unary results for a label (memoised).
    const std::vector<RuleResult>& unary(CatId cat);

    // Eisner NF: may `rr` be built from children produced as described?
    bool nf_allows(const RuleResult& rr, NfKind left_kind, int left_deg, NfKind right_kind, int right_deg) const;

    bool is_root(CatId cat) const;

private:
    std::vector<RuleResult> compute_binary(CatId l, CatId r);
    std::vector<RuleResult> compute_unary(CatId c);
    bool seen(CatId l, CatId r);
    static int rule_degree(Rule r);

    CategoryStore& store_;
    RuleConfig cfg_;
    std::unordered_set<std::string> left_punct_, right_punct_;
    std::vector<std::pair<CatId, CatId>> tc_rules_, tr_rules_;
    std::vector<CatId> roots_;
    std::unordered_set<std::uint64_t> seen_rules_;

    mutable std::shared_mutex mu_;
    std::unordered_map<std::uint64_t, std::vector<RuleResult>> binary_cache_;
    std::unordered_map<CatId, std::vector<RuleResult>> unary_cache_;
};

}  // namespace ccg
