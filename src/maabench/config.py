"""MAABench seed configuration: domains, axes and per-axis counts."""

TARGET_WORDS = (90, 220)

DOMAINS = {
    "news": dict(
        output="news article", corpus="news reports", body="report text", marker="Report",
        fields=["headline"], labels={"headline": "Headline"}),
    "wiki": dict(
        output="encyclopedia article", corpus="encyclopedia entries", body="entry text", marker="Entry",
        fields=["title"], labels={"title": "Title"}),
    "academic": dict(
        output="research abstract", corpus="research paper abstracts", body="abstract", marker="Abstract",
        fields=["title", "field"], labels={"title": "Title", "field": "Field"}),
    "comment": dict(
        output="product review", corpus="customer reviews", body="review text", marker="Review",
        fields=["product", "category", "rating"],
        labels={"product": "Product", "category": "Category", "rating": "Rating"}),
    "patent": dict(
        output="patent abstract", corpus="patent application abstracts", body="abstract", marker="Abstract",
        fields=["title", "classification"],
        labels={"title": "Title", "classification": "Classification"}),
    "stackexchange": dict(
        output="forum question", corpus="questions posted to online Q&A communities", body="question body",
        marker="Question",
        fields=["title", "tags", "community"],
        labels={"title": "Title", "tags": "Tags", "community": "Community"}),
}


COUNTS = dict(
    id_train=30_000,
    id_val=2_000,
    id_test=2_000,
    domain=1_500,
    version=3_000,
    joint=3_000,
)


ID_DOMAINS  = ["news", "wiki", "academic", "comment"]
OOD_DOMAINS = ["patent", "stackexchange"]
TEST_ROLES  = ["test-1", "test-2", "test-3"]

AXES = {
    "maabench_id":       dict(domains=ID_DOMAINS,  roles=["train"],
                           splits={"train": COUNTS["id_train"], "val": COUNTS["id_val"],
                                   "test": COUNTS["id_test"]}),
    "maabench_domain":   dict(domains=OOD_DOMAINS, roles=["train"], total=COUNTS["domain"]),
    "maabench_version":  dict(domains=ID_DOMAINS,  roles=TEST_ROLES, total=COUNTS["version"]),
    "maabench_joint":    dict(domains=OOD_DOMAINS, roles=TEST_ROLES, total=COUNTS["joint"]),
}
