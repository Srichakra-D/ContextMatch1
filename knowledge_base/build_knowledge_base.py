#!/usr/bin/env python3
"""Build the historical entity knowledge base from the full candidate dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
REFERENCE_DATE = "2026-06-20"
RESEARCH_DATE = "2026-06-22"

FICTIONAL_COMPANIES = {
    "Acme Corp",
    "Dunder Mifflin",
    "Globex Inc",
    "Hooli",
    "Initech",
    "Pied Piper",
    "Stark Industries",
    "Wayne Enterprises",
}

# Broad concepts and job functions do not have a single defensible invention date.
NOT_DATEABLE_TECHNOLOGIES = {
    "ASR",
    "Accounting",
    "Agile",
    "BM25",
    "CI/CD",
    "CNN",
    "Computer Vision",
    "Content Matching",
    "Content Writing",
    "Data Pipelines",
    "Data Science",
    "Deep Learning",
    "Diffusion Models",
    "Document Processing",
    "ETL",
    "Embeddings",
    "Feature Engineering",
    "Fine-tuning LLMs",
    "Forecasting",
    "GANs",
    "Image Classification",
    "Indexing Algorithms",
    "Information Retrieval",
    "Information Retrieval Systems",
    "LLMs",
    "Learning to Rank",
    "MLOps",
    "Machine Learning",
    "Marketing",
    "Microservices",
    "Model Adaptation",
    "NLP",
    "Natural Language Processing",
    "Object Detection",
    "Open-source ML libraries",
    "Project Management",
    "Prompt Engineering",
    "RAG",
    "REST APIs",
    "Ranking Systems",
    "Recommendation Systems",
    "Reinforcement Learning",
    "SEO",
    "SQL",
    "Sales",
    "Scrum",
    "Search & Discovery",
    "Search Backend",
    "Search Infrastructure",
    "Semantic Search",
    "Six Sigma",
    "Speech Recognition",
    "Statistical Modeling",
    "TTS",
    "Text Encoders",
    "Time Series",
    "Vector Representations",
    "Vector Search",
    "Workflow Orchestration",
}

# Curated facts are intentionally conservative. Dates mean earliest defensible
# public availability, not current branding or a later major release.
COMPANY_FACTS: dict[str, dict[str, Any]] = {
    "Sarvam AI": {
        "date": "2023",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://www.sarvam.ai/blog/announcing-sarvam-ai",
            "title": "Announcing Sarvam AI",
            "publisher": "Sarvam AI",
            "source_type": "official_company_announcement",
        },
    },
    "Krutrim": {
        "date": "2023",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://www.olakrutrim.com/",
            "title": "Krutrim",
            "publisher": "Krutrim",
            "source_type": "official_company_page",
        },
        "status": "ambiguous",
        "notes": "Official site identifies the company but does not provide a stable, explicit founding date on the referenced page.",
    },
    "Glance": {
        "date": "2019",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://glance.com/about-us/",
            "title": "About Glance",
            "publisher": "Glance",
            "source_type": "official_company_page",
        },
    },
    "Rephrase.ai": {
        "date": "2019",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://www.rephrase.ai/",
            "title": "Rephrase.ai",
            "publisher": "Rephrase.ai",
            "source_type": "official_company_page",
        },
        "status": "ambiguous",
        "notes": "Date requires confirmation from an explicit official history or incorporation source.",
    },
    "CRED": {
        "date": "2018",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://cred.club/about",
            "title": "About CRED",
            "publisher": "CRED",
            "source_type": "official_company_page",
        },
    },
    "Wysa": {
        "date": "2015",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://www.wysa.com/about",
            "title": "About Wysa",
            "publisher": "Wysa",
            "source_type": "official_company_page",
        },
    },
    "Yellow.ai": {
        "date": "2016",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://yellow.ai/about-us/",
            "title": "About Yellow.ai",
            "publisher": "Yellow.ai",
            "source_type": "official_company_page",
        },
    },
    "Observe.AI": {
        "date": "2017",
        "precision": "year",
        "basis": "company_founding",
        "source": {
            "url": "https://www.observe.ai/about-us",
            "title": "About Observe.AI",
            "publisher": "Observe.AI",
            "source_type": "official_company_page",
        },
    },
}

TECHNOLOGY_FACTS: dict[str, dict[str, Any]] = {
    "Angular": ("2010-10", "month", "first_public_release", "https://github.com/angular/angular.js/releases", "AngularJS releases", "Angular"),
    "Apache Beam": ("2016-01", "month", "project_inception", "https://beam.apache.org/blog/2016/01/20/dataflow-beam-and-spark-comparison.html", "Apache Beam project", "Apache Beam"),
    "Apache Flink": ("2011", "year", "project_inception", "https://flink.apache.org/what-is-flink/flink-architecture/", "Apache Flink", "Apache Flink"),
    "BentoML": ("2019", "year", "first_public_release", "https://github.com/bentoml/BentoML/releases", "BentoML releases", "BentoML"),
    "BigQuery": ("2011-11", "month", "public_launch", "https://cloud.google.com/blog/products/data-analytics/google-bigquery-for-big-data-analytics", "Google BigQuery public launch", "Google Cloud"),
    "Databricks": ("2013", "year", "company_product_start", "https://www.databricks.com/company/about-us", "About Databricks", "Databricks"),
    "Django": ("2005-07", "month", "first_public_release", "https://www.djangoproject.com/weblog/2005/jul/13/initial-release/", "Initial Django release", "Django"),
    "Docker": ("2013-03", "month", "first_public_release", "https://www.docker.com/blog/docker-0-1-0/", "Docker 0.1.0", "Docker"),
    "Elasticsearch": ("2010-02", "month", "first_public_release", "https://www.elastic.co/blog/you-know-for-search", "You Know, for Search", "Elastic"),
    "FAISS": ("2017-02", "month", "first_public_release", "https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/", "Faiss similarity search library", "Meta Engineering"),
    "FastAPI": ("2018-12", "month", "first_public_release", "https://github.com/fastapi/fastapi/releases/tag/0.1.0", "FastAPI 0.1.0", "FastAPI"),
    "Flask": ("2010-04", "month", "first_public_release", "https://github.com/pallets/flask/releases/tag/0.1", "Flask 0.1", "Pallets"),
    "Go": ("2009-11", "month", "public_launch", "https://go.dev/blog/hello", "Hey! Ho! Let's Go!", "The Go Authors"),
    "GraphQL": ("2015-09", "month", "public_release", "https://engineering.fb.com/2015/09/14/core-infra/graphql-a-data-query-language/", "GraphQL: A data query language", "Meta Engineering"),
    "Hadoop": ("2006", "year", "project_inception", "https://hadoop.apache.org/history.html", "Hadoop history", "Apache Hadoop"),
    "Haystack": ("2020", "year", "first_public_release", "https://github.com/deepset-ai/haystack/releases", "Haystack releases", "deepset"),
    "Hugging Face Transformers": ("2018", "year", "first_public_release", "https://github.com/huggingface/transformers/releases", "Transformers releases", "Hugging Face"),
    "Java": ("1995-05", "month", "public_launch", "https://www.oracle.com/java/technologies/introduction-to-java.html", "Introduction to Java", "Oracle"),
    "Kafka": ("2011", "year", "first_public_release", "https://kafka.apache.org/08/documentation.html", "Apache Kafka documentation", "Apache Kafka"),
    "Kubeflow": ("2017-12", "month", "public_launch", "https://blog.kubeflow.org/kubeflow/2017/12/21/introducing-kubeflow.html", "Introducing Kubeflow", "Kubeflow"),
    "Kubernetes": ("2014-06", "month", "public_launch", "https://kubernetes.io/blog/2015/04/borg-predecessor-to-kubernetes/", "From Borg to Kubernetes", "Kubernetes"),
    "LangChain": ("2022-10", "month", "first_public_release", "https://github.com/langchain-ai/langchain/releases", "LangChain releases", "LangChain"),
    "LlamaIndex": ("2022-11", "month", "first_public_release", "https://github.com/run-llama/llama_index/releases", "LlamaIndex releases", "LlamaIndex"),
    "LoRA": ("2021-06", "month", "paper_publication", "https://arxiv.org/abs/2106.09685", "LoRA: Low-Rank Adaptation of Large Language Models", "Microsoft Research"),
    "MLflow": ("2018-06", "month", "public_launch", "https://www.databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html", "Introducing MLflow", "Databricks"),
    "Milvus": ("2019-10", "month", "first_public_release", "https://github.com/milvus-io/milvus/releases", "Milvus releases", "Milvus"),
    "MongoDB": ("2009", "year", "first_public_release", "https://www.mongodb.com/company", "MongoDB company", "MongoDB"),
    "Next.js": ("2016-10", "month", "public_launch", "https://vercel.com/blog/next", "Introducing Next.js", "Vercel"),
    "Node.js": ("2009-05", "month", "first_public_release", "https://nodejs.org/en/about/previous-releases", "Previous Node.js releases", "Node.js"),
    "OpenCV": ("2000-06", "month", "first_public_release", "https://opencv.org/about/", "About OpenCV", "OpenCV"),
    "OpenSearch": ("2021-04", "month", "project_launch", "https://aws.amazon.com/blogs/opensource/introducing-opensearch/", "Introducing OpenSearch", "AWS"),
    "PEFT": ("2022-12", "month", "first_public_release", "https://github.com/huggingface/peft/releases", "PEFT releases", "Hugging Face"),
    "Pinecone": ("2019", "year", "company_product_start", "https://www.pinecone.io/company/", "About Pinecone", "Pinecone"),
    "PostgreSQL": ("1996-07", "month", "first_release_under_name", "https://www.postgresql.org/docs/current/history.html", "A Brief History of PostgreSQL", "PostgreSQL"),
    "PyTorch": ("2016-09", "month", "first_public_release", "https://github.com/pytorch/pytorch/releases", "PyTorch releases", "PyTorch"),
    "QLoRA": ("2023-05", "month", "paper_publication", "https://arxiv.org/abs/2305.14314", "QLoRA: Efficient Finetuning of Quantized LLMs", "University of Washington"),
    "Qdrant": ("2020", "year", "first_public_release", "https://github.com/qdrant/qdrant/releases", "Qdrant releases", "Qdrant"),
    "React": ("2013-05", "month", "public_launch", "https://legacy.reactjs.org/blog/2013/06/05/why-react.html", "Why did we build React?", "Meta"),
    "Redis": ("2009", "year", "first_public_release", "https://github.com/redis/redis/releases", "Redis releases", "Redis"),
    "Redux": ("2015-06", "month", "first_public_release", "https://github.com/reduxjs/redux/releases", "Redux releases", "Redux"),
    "Rust": ("2010-07", "month", "public_announcement", "https://blog.mozilla.org/research/2010/07/07/rust/", "Rust", "Mozilla Research"),
    "Sentence Transformers": ("2019", "year", "first_public_release", "https://github.com/UKPLab/sentence-transformers/releases", "Sentence Transformers releases", "UKP Lab"),
    "Snowflake": ("2014", "year", "public_launch", "https://www.snowflake.com/en/company/overview/about-snowflake/", "About Snowflake", "Snowflake"),
    "Spark": ("2009", "year", "project_inception", "https://spark.apache.org/history.html", "Apache Spark history", "Apache Spark"),
    "Spring Boot": ("2014-04", "month", "first_general_release", "https://spring.io/blog/2014/04/01/spring-boot-1-0-ga-released", "Spring Boot 1.0 GA released", "Spring"),
    "Tailwind": ("2017-11", "month", "first_public_release", "https://github.com/tailwindlabs/tailwindcss/releases/tag/v0.1.0", "Tailwind CSS v0.1.0", "Tailwind Labs"),
    "TensorFlow": ("2015-11", "month", "public_launch", "https://opensource.googleblog.com/2015/11/tensorflow-googles-latest-machine_9.html", "TensorFlow open source announcement", "Google"),
    "Terraform": ("2014-07", "month", "public_launch", "https://www.hashicorp.com/blog/terraform", "Terraform", "HashiCorp"),
    "TypeScript": ("2012-10", "month", "public_launch", "https://devblogs.microsoft.com/typescript/announcing-typescript-1-0/", "Announcing TypeScript 1.0", "Microsoft"),
    "Vue.js": ("2014-02", "month", "first_public_release", "https://github.com/vuejs/core/releases", "Vue releases", "Vue.js"),
    "Weaviate": ("2019", "year", "first_public_release", "https://github.com/weaviate/weaviate/releases", "Weaviate releases", "Weaviate"),
    "Weights & Biases": ("2017", "year", "company_product_start", "https://wandb.ai/site/company", "Weights & Biases company", "Weights & Biases"),
    "Webpack": ("2012-03", "month", "first_public_release", "https://github.com/webpack/webpack/releases", "webpack releases", "webpack"),
    "YOLO": ("2015-06", "month", "first_public_release", "https://pjreddie.com/darknet/yolo/", "YOLO: Real-Time Object Detection", "Joseph Redmon"),
    "dbt": ("2016", "year", "first_public_release", "https://www.getdbt.com/blog/dbt-history", "The history of dbt", "dbt Labs"),
    "gRPC": ("2015-02", "month", "public_launch", "https://grpc.io/blog/principles/", "Why gRPC?", "gRPC"),
    "pgvector": ("2021-04", "month", "first_public_release", "https://github.com/pgvector/pgvector/releases", "pgvector releases", "pgvector"),
    "scikit-learn": ("2007", "year", "project_inception", "https://scikit-learn.org/stable/about.html", "About scikit-learn", "scikit-learn"),
}

# Convert compact tuples above to normal fact dictionaries.
TECHNOLOGY_FACTS = {
    name: {
        "date": values[0],
        "precision": values[1],
        "basis": values[2],
        "source": {
            "url": values[3],
            "title": values[4],
            "publisher": values[5],
            "source_type": (
                "original_research_paper"
                if "arxiv.org" in values[3]
                else "official_documentation"
            ),
        },
    }
    for name, values in TECHNOLOGY_FACTS.items()
}

CERTIFICATION_FACTS: dict[str, dict[str, Any]] = {
    "AWS Certified Cloud Practitioner|AWS": {
        "date": "2018",
        "precision": "year",
        "basis": "certification_launch",
        "source": {
            "url": "https://aws.amazon.com/blogs/apn/aws-certified-cloud-practitioner-exam-now-available/",
            "title": "AWS Certified Cloud Practitioner Exam Now Available",
            "publisher": "AWS",
            "source_type": "official_provider_announcement",
        },
    },
    "AWS Certified Machine Learning Specialty|AWS": {
        "date": "2019",
        "precision": "year",
        "basis": "certification_launch",
        "source": {
            "url": "https://aws.amazon.com/blogs/training-and-certification/aws-certified-machine-learning-specialty-exam-now-available/",
            "title": "AWS Certified Machine Learning Specialty Exam Now Available",
            "publisher": "AWS",
            "source_type": "official_provider_announcement",
        },
    },
    "Google Cloud Professional ML Engineer|Google Cloud": {
        "date": "2020",
        "precision": "year",
        "basis": "certification_launch",
        "source": {
            "url": "https://cloud.google.com/blog/products/ai-machine-learning/google-cloud-launches-machine-learning-engineer-certification",
            "title": "Google Cloud launches Professional Machine Learning Engineer certification",
            "publisher": "Google Cloud",
            "source_type": "official_provider_announcement",
        },
    },
    "LangChain for LLM Application Development|DeepLearning.AI": {
        "date": "2023",
        "precision": "year",
        "basis": "course_launch",
        "source": {
            "url": "https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/",
            "title": "LangChain for LLM Application Development",
            "publisher": "DeepLearning.AI",
            "source_type": "official_provider_page",
        },
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(path: Path) -> tuple[int, Counter, Counter, Counter]:
    companies: Counter[str] = Counter()
    technologies: Counter[str] = Counter()
    certifications: Counter[str] = Counter()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            candidate = json.loads(line)
            count += 1
            companies.update(role["company"] for role in candidate["career_history"])
            technologies.update(skill["name"] for skill in candidate.get("skills", []))
            certifications.update(
                f"{certification['name']}|{certification['issuer']}"
                for certification in candidate.get("certifications", [])
            )
    return count, companies, technologies, certifications


def source_entry(source: dict[str, str]) -> dict[str, str]:
    return {**source, "retrieved_on": RESEARCH_DATE}


def make_unknown(name: str, occurrences: int, entity_type: str) -> dict[str, Any]:
    return {
        "canonical_name": name,
        "aliases": [],
        "status": "unknown",
        "occurrences": occurrences,
        "date": None,
        "date_precision": None,
        "date_basis": None,
        "sources": [],
        "notes": f"No primary-source historical date has been approved for this {entity_type}.",
    }


def make_fact(name: str, occurrences: int, fact: dict[str, Any]) -> dict[str, Any]:
    status = fact.get("status", "verified")
    return {
        "canonical_name": name,
        "aliases": fact.get("aliases", []),
        "status": status,
        "occurrences": occurrences,
        "date": fact["date"],
        "date_precision": fact["precision"],
        "date_basis": fact["basis"],
        "sources": [source_entry(fact["source"])],
        "notes": fact.get("notes", ""),
    }


def build(source_path: Path) -> dict[str, Any]:
    candidate_count, companies, technologies, certifications = inventory(source_path)

    company_entries = {}
    for name in sorted(companies):
        if name in FICTIONAL_COMPANIES:
            company_entries[name] = {
                "canonical_name": name,
                "aliases": [],
                "status": "fictional",
                "occurrences": companies[name],
                "date": None,
                "date_precision": None,
                "date_basis": None,
                "sources": [],
                "notes": "Synthetic placeholder used throughout the generated dataset; the name alone is not an integrity failure.",
            }
        elif name in COMPANY_FACTS:
            company_entries[name] = make_fact(name, companies[name], COMPANY_FACTS[name])
        else:
            company_entries[name] = make_unknown(name, companies[name], "company")

    technology_entries = {}
    for name in sorted(technologies):
        if name in NOT_DATEABLE_TECHNOLOGIES:
            technology_entries[name] = {
                "canonical_name": name,
                "aliases": [],
                "status": "not_dateable",
                "occurrences": technologies[name],
                "date": None,
                "date_precision": None,
                "date_basis": None,
                "sources": [],
                "notes": "Broad concept or practice without one defensible first-availability date.",
            }
        elif name in TECHNOLOGY_FACTS:
            technology_entries[name] = make_fact(
                name, technologies[name], TECHNOLOGY_FACTS[name]
            )
        else:
            technology_entries[name] = make_unknown(
                name, technologies[name], "technology"
            )

    certification_entries = {}
    for key in sorted(certifications):
        if key in CERTIFICATION_FACTS:
            certification_entries[key] = make_fact(
                key, certifications[key], CERTIFICATION_FACTS[key]
            )
        else:
            certification_entries[key] = make_unknown(
                key, certifications[key], "certification"
            )

    return {
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "reference_date": REFERENCE_DATE,
            "generated_on": RESEARCH_DATE,
            "research_cutoff_date": RESEARCH_DATE,
            "source_dataset": str(source_path),
            "source_dataset_sha256": sha256(source_path),
            "source_candidate_count": candidate_count,
            "company_count": len(company_entries),
            "technology_count": len(technology_entries),
            "certification_count": len(certification_entries),
            "verified_facts_require_primary_sources": True,
        },
        "companies": company_entries,
        "technologies": technology_entries,
        "certifications": certification_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("India_runs_data_and_ai_challenge/candidates.jsonl"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("knowledge_base.json")
    )
    args = parser.parse_args()
    result = build(args.source)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {args.output}: "
        f"{result['metadata']['company_count']} companies, "
        f"{result['metadata']['technology_count']} technologies, "
        f"{result['metadata']['certification_count']} certifications."
    )


if __name__ == "__main__":
    main()
