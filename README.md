# Programming for AI

Landing page for group **G7**, course Programming Foundation for Data Analysis and Visualization (Semester 261, academic year 2026–2027).

The site is published with GitHub Pages from `docs/`:

https://tatkien.github.io/programming-for-ai/

## Team

- Nguyen Tat Kien — 2311735
- Nguyen Thanh Minh Khoi — 2311687
- Nguyen Lam Huy — 2311188

Roles and GitHub profile links are edited in [docs/content/team.json](docs/content/team.json).

## Projects

1. **Tabular** — edit [docs/content/tabular.md](docs/content/tabular.md). Figures go in `docs/assets/tabular/`.
2. **Text** — edit [docs/content/text.md](docs/content/text.md). Figures go in `docs/assets/text/`.
3. **Image** — edit [docs/content/image.md](docs/content/image.md). Figures go in `docs/assets/image/`.

Set `status: ready` at the top of a project file when that part is ready for the lecturer. Leave it as `draft` while the section still says "Not yet updated". Image paths in Markdown are relative to `docs/` (for example `assets/image/figure.png`). Colab, notebook, PDF, and YouTube links are the `colab`, `notebook`, `pdf`, and `video` fields in that same file.

Notebooks stay in the repository root.

To preview locally, run `python -m http.server` from the `docs/` folder, then open http://127.0.0.1:8000/. Opening the HTML files directly will not load the Markdown.
