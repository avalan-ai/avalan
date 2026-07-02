# Skills Examples

This directory contains hermetic skill packages used by documentation and
tests. They are trusted only when an operator, SDK, or host process points a
skills source at this directory.

- [pdf/SKILL.md](pdf/SKILL.md) teaches a PDF workflow. It is read through
  logical skill ID `pdf`, not through a host path.

Example agent run:

```bash
echo "Use the PDF skill to plan a review." \
  | avalan agent run docs/examples/agent_skills_pdf.toml \
      --tool-skills-source workspace-main=docs/examples/skills \
      --tool-skills-source-authority workspace-main=workspace:docs \
      --display-tools
```
