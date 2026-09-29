"use strict";

(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
  root.G7 = api;
  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", function () {
      api.boot();
    });
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function escapeHtml(value) {
    return String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function escapeAttr(value) {
    return escapeHtml(value).replace(/"/g, "&quot;");
  }

  function safeUrl(url) {
    const value = String(url || "").trim();
    if (!value || /[\s<>"']/.test(value)) return false;
    if (/^(javascript|data|vbscript):/i.test(value)) return false;
    if (value.startsWith("//") || value.startsWith("/")) return false;
    if (value.split("/").includes("..")) return false;
    if (/^https?:\/\//i.test(value)) return true;
    if (/^mailto:/i.test(value)) return true;
    if (value.startsWith("#")) return true;
    return /^[A-Za-z0-9._~-]/.test(value);
  }

  function inline(text) {
    const tokens = [];
    function hold(html) {
      tokens.push(html);
      return "\u0000" + (tokens.length - 1) + "\u0000";
    }

    let source = String(text);
    source = source.replace(/!\[([^\]]*)\]\(([^)\s]+)\)/g, function (full, alt, url) {
      if (!safeUrl(url)) return full;
      return hold('<img src="' + escapeAttr(url) + '" alt="' + escapeAttr(alt) + '">');
    });
    source = source.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (full, label, url) {
      if (!safeUrl(url)) return full;
      const external = /^https?:\/\//i.test(url) ? ' target="_blank" rel="noopener noreferrer"' : "";
      return hold('<a href="' + escapeAttr(url) + '"' + external + ">" + escapeHtml(label) + "</a>");
    });
    source = source.replace(/`([^`]+)`/g, function (full, code) {
      return hold("<code>" + escapeHtml(code) + "</code>");
    });
    source = source.replace(/\*\*([^*]+)\*\*/g, function (full, bold) {
      return hold("<strong>" + escapeHtml(bold) + "</strong>");
    });

    return escapeHtml(source).replace(/\u0000(\d+)\u0000/g, function (_, id) {
      return tokens[Number(id)];
    });
  }

  function splitRow(line) {
    let source = line.trim();
    if (source.startsWith("|")) source = source.slice(1);
    if (source.endsWith("|")) source = source.slice(0, -1);
    return source.split("|").map(function (cell) {
      return cell.trim();
    });
  }

  function isSeparator(line) {
    if (!line || line.indexOf("|") === -1 || line.indexOf("-") === -1) return false;
    const cells = splitRow(line);
    return cells.length > 0 && cells.every(function (cell) {
      return /^:?-{3,}:?$/.test(cell);
    });
  }

  function isSpecial(line, next) {
    if (!line) return false;
    if (line.startsWith("```")) return true;
    if (/^#{1,3} /.test(line)) return true;
    if (/^[-*] /.test(line)) return true;
    if (line.indexOf("|") !== -1 && isSeparator(next)) return true;
    return false;
  }

  function renderMarkdown(src) {
    const lines = String(src || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n").split("\n");
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index];
      if (line.trim() === "") {
        index += 1;
        continue;
      }

      if (line.startsWith("```")) {
        const buffer = [];
        index += 1;
        while (index < lines.length && !lines[index].startsWith("```")) {
          buffer.push(lines[index]);
          index += 1;
        }
        if (index < lines.length) index += 1;
        blocks.push("<pre><code>" + escapeHtml(buffer.join("\n")) + "</code></pre>");
        continue;
      }

      const heading = line.match(/^(#{1,3}) (.+)$/);
      if (heading) {
        const level = heading[1].length;
        blocks.push("<h" + level + ">" + inline(heading[2]) + "</h" + level + ">");
        index += 1;
        continue;
      }

      if (line.indexOf("|") !== -1 && isSeparator(lines[index + 1])) {
        const headers = splitRow(line);
        index += 2;
        const rows = [];
        while (index < lines.length && lines[index].trim() !== "" && lines[index].indexOf("|") !== -1) {
          rows.push(splitRow(lines[index]));
          index += 1;
        }
        const head = "<tr>" + headers.map(function (cell) {
          return "<th>" + inline(cell) + "</th>";
        }).join("") + "</tr>";
        const body = rows.map(function (row) {
          const cells = headers.map(function (_, cellIndex) {
            return "<td>" + inline(row[cellIndex] || "") + "</td>";
          });
          return "<tr>" + cells.join("") + "</tr>";
        }).join("");
        blocks.push("<table><thead>" + head + "</thead><tbody>" + body + "</tbody></table>");
        continue;
      }

      if (/^[-*] /.test(line)) {
        const items = [];
        while (index < lines.length && /^[-*] /.test(lines[index])) {
          items.push("<li>" + inline(lines[index].replace(/^[-*] /, "")) + "</li>");
          index += 1;
        }
        blocks.push("<ul>" + items.join("") + "</ul>");
        continue;
      }

      const paragraph = [];
      while (index < lines.length && lines[index].trim() !== "" && !isSpecial(lines[index], lines[index + 1])) {
        paragraph.push(lines[index].trim());
        index += 1;
      }
      blocks.push("<p>" + inline(paragraph.join(" ")) + "</p>");
    }

    return blocks.join("\n");
  }

  function parseFrontMatter(text) {
    const src = String(text || "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
    const frontMatter = {};
    let body = src;
    if (src.startsWith("---\n")) {
      const close = src.indexOf("\n---", 4);
      if (close !== -1) {
        src.slice(4, close).split("\n").forEach(function (line) {
          const match = line.match(/^([A-Za-z0-9_-]+):\s*(.*)$/);
          if (!match) return;
          let value = match[2].trim();
          const quoted = (value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"));
          if (quoted && value.length >= 2) value = value.slice(1, -1);
          frontMatter[match[1]] = value;
        });
        body = src.slice(close + 4);
        if (body.startsWith("\n")) body = body.slice(1);
      }
    }
    return { frontMatter: frontMatter, body: body };
  }

  function parseDocument(text) {
    const parsed = parseFrontMatter(text);
    return {
      frontMatter: parsed.frontMatter,
      html: renderMarkdown(parsed.body)
    };
  }

  function statusReady(status) {
    return String(status || "").trim().toLowerCase() === "ready";
  }

  function applyStatus(el, status) {
    const ready = statusReady(status);
    el.textContent = ready ? "Ready to submit" : "In progress";
    el.classList.toggle("ready", ready);
    el.dataset.status = ready ? "ready" : "draft";
  }

  function linkOrPending(url, label) {
    const value = String(url || "").trim();
    if (!safeUrl(value)) return '<span class="pending">Not yet updated</span>';
    return '<a href="' + escapeAttr(value) + '" target="_blank" rel="noopener noreferrer">' + escapeHtml(label) + "</a>";
  }

  function setText(selector, value) {
    const el = document.querySelector(selector);
    if (el) el.textContent = value;
  }

  function setHtml(selector, value) {
    const el = document.querySelector(selector);
    if (el) el.innerHTML = value;
  }

  function fillTeam() {
    return fetch("content/team.json")
      .then(function (res) {
        if (!res.ok) throw new Error(String(res.status));
        return res.json();
      })
      .then(function (team) {
        document.querySelectorAll("[data-group-name]").forEach(function (el) {
          el.textContent = team.group || "G7";
        });
        document.querySelectorAll("[data-repo-link]").forEach(function (el) {
          if (team.repo) el.href = team.repo;
        });
        document.querySelectorAll("[data-member-list]").forEach(function (list) {
          list.innerHTML = (team.members || []).map(function (member) {
            const github = safeUrl(member.github)
              ? ' — <a href="' + escapeAttr(member.github) + '" target="_blank" rel="noopener noreferrer">GitHub</a>'
              : "";
            return "<li><strong>" + escapeHtml(member.name) + "</strong> — " + escapeHtml(member.id) + " — <span class=\"role\">" + escapeHtml(member.role || "Not yet updated") + "</span>" + github + "</li>";
          }).join("");
        });
      })
      .catch(function () {});
  }

  function fillHome() {
    [
      ["tabular", "content/tabular.md"],
      ["text", "content/text.md"],
      ["image", "content/image.md"]
    ].forEach(function (item) {
      const id = item[0];
      const file = item[1];
      fetch(file)
        .then(function (res) {
          if (!res.ok) throw new Error(String(res.status));
          return res.text();
        })
        .then(function (text) {
          const frontMatter = parseFrontMatter(text).frontMatter;
          const card = document.querySelector('[data-project="' + id + '"]');
          if (!card) return;
          const summary = card.querySelector(".summary");
          if (summary) summary.textContent = frontMatter.summary || "Not yet updated";
          const pill = card.querySelector(".pill");
          if (pill) applyStatus(pill, frontMatter.status);
        })
        .catch(function () {});
    });
  }

  function fillProject() {
    const file = document.body.dataset.content;
    if (!file) return;
    const writeup = document.querySelector("#writeup");
    fetch(file)
      .then(function (res) {
        if (!res.ok) throw new Error(String(res.status));
        return res.text();
      })
      .then(function (text) {
        const doc = parseDocument(text);
        const fm = doc.frontMatter;
        setText('[data-field="problem"]', fm.problem || "Not yet updated");
        setText('[data-field="dataset"]', fm.dataset || "Not yet updated");
        setHtml('[data-field="colab"]', linkOrPending(fm.colab, "Open Colab"));
        setHtml('[data-field="notebook"]', linkOrPending(fm.notebook, "Open notebook"));
        setHtml('[data-field="pdf"]', linkOrPending(fm.pdf, "Download PDF"));
        setHtml('[data-field="video"]', linkOrPending(fm.video, "Watch video"));
        const pill = document.querySelector("[data-status-pill]");
        if (pill) applyStatus(pill, fm.status);
        if (writeup) writeup.innerHTML = doc.html || "<p>Not yet updated.</p>";
      })
      .catch(function () {
        if (writeup) writeup.innerHTML = "<p>Could not load this section.</p>";
      });
  }

  function boot() {
    fillTeam();
    if (document.body.dataset.page === "home") fillHome();
    if (document.body.dataset.page === "project") fillProject();
  }

  return {
    parseFrontMatter: parseFrontMatter,
    parseDocument: parseDocument,
    renderMarkdown: renderMarkdown,
    safeUrl: safeUrl,
    linkOrPending: linkOrPending,
    statusReady: statusReady,
    boot: boot
  };
});
