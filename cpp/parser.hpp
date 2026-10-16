// Viterbi CKY over a supertag lattice.
//
// Each token carries a list of (category string, log-probability) candidates.
// The chart keeps, per span, the best-scoring item for every (category,
// NF-class) key; the score of an item is the sum of the lexical
// log-probabilities of its leaves (plus optional unary penalties). The best
// derivation whose root matches a root category is returned.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "category.hpp"
#include "grammar.hpp"

namespace ccg {

struct LexCandidate {
    std::string text;                   // original supertag string (may contain -->)
    std::vector<std::string> segments;  // text split on "-->"
    std::vector<CatId> chain;           // parsed segments (leaf first)
    CatId cat = kNoCat;                 // final category (last chain element)
    bool valid = false;
};

// Shared, thread-safe cache of supertag string -> LexCandidate.
class Lexicon {
public:
    explicit Lexicon(CategoryStore& store) : store_(store) {}
    const LexCandidate& lookup(const std::string& text);

private:
    CategoryStore& store_;
    mutable std::shared_mutex mu_;
    std::unordered_map<std::string, LexCandidate> cache_;
};

struct Node {
    std::string cat;         // category label of this node
    std::string rule;        // rule name
    int head = 0;
    int start = 0, end = 0;  // half-open token span
    std::vector<int> children;
    // leaves only
    int token = -1;
    int cand = -1;
    std::string word;
    std::string supertag;    // original supertag string of the candidate
    double logp = 0.0;
};

struct ParseOutput {
    bool ok = false;
    double score = 0.0;
    std::string root_cat;
    std::vector<Node> nodes;
    int root = -1;
    std::vector<int> chosen;       // per token: candidate index used by the derivation (-1 if failed)
    std::size_t chart_items = 0;
    std::size_t root_items = 0;    // items of any category in the full-span cell
    std::vector<std::string> root_cell_cats;  // full-span cell labels (debugging failed parses)
};

using Lattice = std::vector<std::vector<std::pair<std::string, double>>>;

class Parser {
public:
    Parser(CategoryStore& store, Grammar& grammar, Lexicon& lexicon);
    ParseOutput parse(const std::vector<std::string>& tokens, const Lattice& lattice);

private:
    struct Item {
        CatId cat = kNoCat;
        double score = 0.0;
        // Secondary key on score ties (lower wins): sum over punctuation
        // absorptions of (n - span length), so punctuation attaches as high
        // as possible, as in CCGbank.
        std::int32_t tie = 0;
        Rule rule = Rule::Lex;
        std::uint8_t head = 0;
        NfKind nf_kind = NfKind::Other;
        std::uint8_t nf_degree = 0;
        std::uint8_t unary_depth = 0;
        std::int32_t left = -1;    // pool index
        std::int32_t right = -1;   // pool index
        std::int16_t start = 0, end = 0;
        std::int32_t cand = -1;    // leaves: index into the token's candidate list
    };

    struct Cell {
        std::vector<std::int32_t> items;
        std::unordered_map<std::uint64_t, std::int32_t> index;  // (cat, nf) -> pool index
    };

    static std::uint64_t item_key(CatId cat, NfKind kind, int degree);
    static bool better(const Item& candidate, const Item& incumbent);
    bool insert(Cell& cell, Item item);
    void unary_closure(Cell& cell);
    void finalize_cell(Cell& cell);
    int emit(const std::vector<std::string>& tokens, const Lattice& lattice, std::int32_t idx, ParseOutput& out);

    CategoryStore& store_;
    Grammar& grammar_;
    Lexicon& lexicon_;
    std::vector<Item> pool_;
};

}  // namespace ccg
