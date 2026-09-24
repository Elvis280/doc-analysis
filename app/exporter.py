import os
import json
import zipfile
import shutil
from typing import List
from app.schemas import KnowledgeUnit, Manifest


def export_kb(units: List[KnowledgeUnit], pdf_filename: str, output_dir: str) -> str:
    """
    Exports knowledge units to Markdown and JSON formats, creates a manifest,
    and returns the path to a ZIP file.

    Args:
        units:        The extracted knowledge units.
        pdf_filename: Original PDF filename (used for naming).
        output_dir:   Per-job directory (already created by the caller).
                      Each job gets its own UUID-namespaced output_dir so
                      concurrent runs never clobber each other.
    """
    kb_name = os.path.splitext(pdf_filename)[0] + "_kb"
    kb_dir = os.path.join(output_dir, kb_name)
    knowledge_dir = os.path.join(kb_dir, "knowledge")
    metadata_dir = os.path.join(kb_dir, "metadata")

    os.makedirs(knowledge_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Export units
    for unit in units:
        # Markdown export
        md_path = os.path.join(knowledge_dir, f"{unit.id}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# {unit.title}\n\n")
            f.write(f"**Type:** {unit.type.value}\n")
            if unit.section:
                f.write(f"**Section:** {unit.section}\n")
            f.write(f"**Keywords:** {', '.join(unit.keywords)}\n\n")
            f.write(unit.content)

            if unit.relationships:
                f.write("\n\n## Relationships\n")
                for rel in unit.relationships:
                    f.write(f"- **{rel.type.value}**: {rel.target_id}\n")

        # JSON metadata export
        json_path = os.path.join(metadata_dir, f"{unit.id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(unit.model_dump_json(indent=2))

    # Create manifest
    manifest = Manifest(
        source_pdf=pdf_filename,
        kb_name=kb_name,
        num_units=len(units),
        has_relationships=any(len(u.relationships) > 0 for u in units),
    )
    manifest_path = os.path.join(kb_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write(manifest.model_dump_json(indent=2))

    # Create ZIP inside the per-job output_dir
    zip_path = os.path.join(output_dir, f"{kb_name}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(kb_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_dir)
                zipf.write(file_path, arcname)

    return zip_path
