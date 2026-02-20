#!/usr/bin/env bash
# =============================================================================
# recover_deleted_code.sh
#
# Restores the three legacy research/UX modules that were removed from the
# working tree to keep the repository lean.  All modules are preserved in full
# in the Git history — this script simply checks them back out.
#
# Run from the repository root:
#
#   bash scripts/recover_deleted_code.sh [module]
#
# Valid module names: code-DRMinerva   code-KG   MongoDB-UI
# Omit the argument to restore ALL three.
#
# Prerequisites:
#   git must be available and the working tree must be inside the original
#   MedPix-2-0 Git repository.
# =============================================================================
set -e
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

restore_from_git() {
  local dir="$1"
  if [ -d "$dir" ] && [ "$(ls -A "$dir" 2>/dev/null)" ]; then
    echo "  $dir already present — skipping."
    return
  fi

  echo "  Restoring $dir from Git history ..."
  # Use the last commit that still contained this directory
  COMMIT=$(git log --diff-filter=D --summary --format="%H" -- "$dir/" 2>/dev/null | head -1)
  if [ -z "$COMMIT" ]; then
    # Fall back: the files might simply be tracked at HEAD (e.g. if this script
    # is run before they are committed away)
    git checkout HEAD -- "$dir/" 2>/dev/null && echo "  Restored $dir from HEAD." && return
    echo "  ERROR: could not find $dir in Git history. Manual recovery needed."
    return 1
  fi
  # Restore from the commit just BEFORE the delete commit
  PARENT="${COMMIT}^"
  git checkout "$PARENT" -- "$dir/"
  echo "  Restored $dir (from $PARENT)."
}

# ── What to restore ──────────────────────────────────────────────────────────
# Descriptions of each removed module:
#
# code-DRMinerva/
#   DR-Minerva: Flamingo-based multimodal LLM fine-tuning + RAG pipeline.
#   Published code for: AIxIA 2024 paper (Siragusa et al., 2025).
#   Execution order: RAG_costruction_history.py → finetune-minerva.py →
#     merge_weights.py → inference_rag_gen.py → gen_dataset_ft_flamingo.py →
#     train.py → eval.py → evaluate-code.py
#   Dependencies: transformers, peft, accelerate, faiss, sentence-transformers
#
# code-KG/
#   Knowledge Graph augmented DR-Minerva pipeline.
#   Published code for: arXiv:2407.02994 (Siragusa et al., 2025).
#   Execution order: gen_template_kg.py → gen_kg.py → gen-questions.py →
#     inference-KG.py → evaluate-csv.py
#   Dependencies: neo4j (or networkx), langchain, sentence-transformers
#
# MongoDB-UI/
#   PyQt5 desktop UI for querying MedPix 2.0 via a local MongoDB instance.
#   Run mongoDB_creation.py once to load data, then mongoDB_inference.py for UI.
#   Dependencies: pymongo, PyQt5, pillow, a running MongoDB server (port 27017)
# =============================================================================

MODULES=("code-DRMinerva" "code-KG" "MongoDB-UI")

if [ -n "$1" ]; then
  VALID=0
  for m in "${MODULES[@]}"; do
    [ "$1" == "$m" ] && VALID=1
  done
  if [ "$VALID" -eq 0 ]; then
    echo "Unknown module: $1"
    echo "Valid choices: ${MODULES[*]}"
    exit 1
  fi
  MODULES=("$1")
fi

echo "========================================"
echo " MedPix-2-0 code module recovery"
echo " Modules: ${MODULES[*]}"
echo "========================================"
for module in "${MODULES[@]}"; do
  restore_from_git "$module"
done

echo ""
echo "Done.  Install extra dependencies per each module's README."
echo ""
echo "Quick dependency hints:"
echo "  code-DRMinerva : pip install transformers peft accelerate faiss-cpu sentence-transformers"
echo "  code-KG        : pip install neo4j langchain sentence-transformers"
echo "  MongoDB-UI     : pip install pymongo PyQt5 pillow   # + start MongoDB on port 27017"
