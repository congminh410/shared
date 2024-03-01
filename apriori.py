def load_data():
    # Replace this with your data loading logic
    return [
        ['bread', 'milk'],
        ['bread', 'diaper', 'beer', 'egg'],
        ['milk', 'diaper', 'beer', 'cola'],
        ['bread', 'milk', 'diaper', 'beer'],
        ['bread', 'milk', 'diaper', 'cola']
    ]

def generate_candidates(itemset, length):
    candidates = []

    if length == 1:
        unique_items = set(item for transaction in itemset for item in transaction)
        return [[item] for item in unique_items]

    for i in range(len(itemset)):
        for j in range(i + 1, len(itemset)):
            candidate = list(set(itemset[i]) | set(itemset[j]))
            if len(candidate) == length:
                candidates.append(sorted(candidate))
    return candidates

def prune(itemset, prev_itemsets):
    '''
    to check whether all the subsets of a candidate itemset are frequent. 
    If any subset of the candidate itemset is not frequent, 
    then the candidate itemset is pruned (removed) from further consideration.
    '''
    for item in itemset:
        subset = itemset.copy()
        subset.remove(item)
        if subset not in prev_itemsets:
            return True
    return False

def apriori(data, min_support):
    itemsets = [list(set(transaction)) for transaction in data]
    itemsets.sort()

    supports = {}
    frequent_itemsets = {}

    length = 1
    while itemsets:
        candidates = generate_candidates(itemsets, length)
        frequent_candidates = []

        for candidate in candidates:
            support = sum(1 for transaction in data if set(candidate).issubset(transaction))
            supports[tuple(candidate)] = support / len(data)

            if supports[tuple(candidate)] >= min_support and prune(candidate, frequent_candidates):
                frequent_candidates.append(candidate)

        frequent_itemsets[length] = frequent_candidates
        itemsets = frequent_candidates
        length += 1

    return supports, frequent_itemsets

# def generate_rules(frequent_itemsets, supports, min_confidence):
#     rules = []

#     for length, itemsets in frequent_itemsets.items():
#         for itemset in itemsets:
#             for i in range(1, len(itemset)):
#                 antecedent = itemset[:i]
#                 consequent = itemset[i:]

#                 confidence = supports[tuple(itemset)] / supports[tuple(antecedent)]
#                 if confidence >= min_confidence:
#                     rule = {
#                         'antecedent': antecedent,
#                         'consequent': consequent,
#                         'confidence': confidence
#                     }
#                     rules.append(rule)

#     return rules

if __name__ == "__main__":
    data = load_data()
    min_support = 0.6
    min_confidence = 0.7

    supports, frequent_itemsets = apriori(data, min_support)
    # rules = generate_rules(frequent_itemsets, supports, min_confidence)

    print("Frequent Itemsets:")
    for length, itemsets in frequent_itemsets.items():
        print(f"Length {length}: {itemsets}")

    # print("\nAssociation Rules:")
    # for rule in rules:
    #     print(f"{rule['antecedent']} -> {rule['consequent']}, Confidence: {rule['confidence']:.2f}")
