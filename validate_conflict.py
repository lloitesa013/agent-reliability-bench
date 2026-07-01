import collections, data_engine as de
class Stub:
    # simulate bge: candidates already ordered [correct(, corrupt), ...fillers]; return top-k
    def top_k(self, query, docs, k=1):
        return docs[:k], [1.0]*k
p = de.build_partials(Stub())
print("total partials:", len(p))
print("by case_type :", dict(collections.Counter(x["case_type"] for x in p)))
print("all have 'group':", all("group" in x for x in p))
dis = [x for x in p if x["case_type"]=="distractor"]
dr  = [x for x in p if x["case_type"]=="direct"]
rs  = [x for x in p if x["case_type"]=="reasoning"]
print("retrieved-len  direct:", dict(collections.Counter(len(x["retrieved"]) for x in dr)),
      " distractor:", dict(collections.Counter(len(x["retrieved"]) for x in dis)),
      " reasoning:", dict(collections.Counter(len(x["retrieved"]) for x in rs)))
print()
print("=== 3 distractor partials (CONFLICT = 2 docs) ===")
for x in dis[:3]:
    print("q :", x["query"])
    for i,d in enumerate(x["retrieved"]): print("   doc%d: %s"%(i,d))
    print("   group:", x["group"])
# concern-2: direct & distractor of same fact must co-group
gd = set(x["group"] for x in dr); gx = set(x["group"] for x in dis)
print()
print("concern-2 splittable: direct groups=%d distractor groups=%d shared=%d"%(len(gd),len(gx),len(gd & gx)))
print("total distinct groups:", len(set(x["group"] for x in p)))
