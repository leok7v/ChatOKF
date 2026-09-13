import Foundation

public enum StoreText {

    public static func estimatedTokens(bytes count: Int) -> Int {
        max(1, (count + 3) / 4)
    }

    public static func flags(_ concept: Concept) -> String {
        var out = concept.isStale ? " [stale]" : ""
        if concept.status == "deprecated" { out += " [deprecated]" }
        if concept.trust != .unverified {
            out += " [" + concept.trust.rawValue + "]"
        }
        return out
    }

    public static func confidence(_ result: SearchResult,
                           _ embedder: Embedder) -> String {
        var out = ""
        let literal = result.hits.first?.terms.isEmpty == false
        let vague = result.standout < embedder.standoutFloor
        if !result.hits.isEmpty && vague && !literal {
            out = "[weak: no clear match; answer from your own "
                + "knowledge rather than from these]\n"
        }
        return out
    }

    public static func map(_ store: Store, ids withIds: Bool) -> String {
        let areas = store.areas()
        var out = ""
        if let owner = store.owner {
            out += "\(owner.title)\n  \(owner.description)\n"
            out += "  full profile: \(owner.id)\n\n"
        }
        let retired = store.concepts.filter { concept in
            concept.isDeprecated
        }.count
        out += "\(store.concepts.count - retired) concepts"
        if retired > 0 { out += " (+\(retired) deprecated)" }
        out += " in \(areas.count) areas."
        if !store.okfVersion.isEmpty {
            out += " OKF \(store.okfVersion)."
        }
        out += " Search by meaning; these are areas, not ids.\n\n"
        let width = areas.map { area in area.area.count }.max() ?? 8
        for area in areas {
            let pad = String(repeating: " ",
                             count: width - area.area.count)
            out += "  \(area.area)/\(pad) \(area.count)  "
            out += area.themes.joined(separator: ", ") + "\n"
            if withIds {
                for id in store.ids(inArea: area.area) {
                    out += "      \(id)\n"
                }
            }
        }
        return out
    }

    public static func search(_ store: Store, _ embedder: Embedder,
                       queries: [String], filter: Filter,
                       limit: Int) -> String {
        let result = store.search(queries, filter: filter, limit: limit)
        var out = confidence(result, embedder)
        if result.hits.isEmpty {
            out += filter.isEmpty
                ? "no concepts indexed\n"
                : "no concept matches that filter\n"
        }
        for hit in result.hits {
            var flags = flags(hit.concept)
            if !hit.terms.isEmpty {
                flags += " [" + hit.terms.joined(separator: " ") + "]"
            }
            out += String(format: "%@  %.3f%@\n", hit.concept.id,
                          hit.score, flags)
            out += "  " + hit.concept.title + "\n"
            if !hit.concept.description.isEmpty {
                out += "  " + hit.concept.description + "\n"
            }
        }
        return out
    }

    public static func read(_ store: Store, _ concept: Concept, about: String?,
                     limit: Int, offset: Int) -> String {
        var out = "# \(concept.title)\n"
        out += "id: \(concept.id)  type: \(concept.type)"
        if !concept.tags.isEmpty {
            out += "  tags: " + concept.tags.joined(separator: ", ")
        }
        if concept.isStale { out += "  [stale]" }
        if concept.trust != .unverified {
            out += "  [" + concept.trust.rawValue + "]"
        }
        if !concept.status.isEmpty {
            out += "  status: \(concept.status)"
        }
        out += "\n\n"
        let bytes = Array(concept.body.utf8)
        var start = min(max(0, offset), bytes.count)
        var end = limit < 0
            ? bytes.count : min(bytes.count, start + limit)
        if let about = about, limit > 0, offset == 0 {
            let span = store.window(concept, about: about, limit: limit)
            start = span.from
            end = span.to
        }
        if start > 0 {
            out += "[\(start) earlier bytes not shown]\n\n"
        }
        out += String(decoding: bytes[start..<end], as: UTF8.self)
        if end < bytes.count {
            out += "\n\n[\(bytes.count - end) more bytes;"
            out += " read \(concept.id) --offset \(end) for them]"
        }
        if !concept.links.isEmpty {
            out += "\n\nlinks: " + concept.links.joined(separator: " ")
        }
        if !concept.backlinks.isEmpty {
            out += "\nbacklinks: "
            out += concept.backlinks.joined(separator: " ")
        }
        return out
    }

    public static func grep(_ hits: [GrepHit]) -> String {
        var out = hits.isEmpty ? "no match\n" : ""
        for hit in hits {
            out += "\(hit.id):\(hit.line)  \(hit.text)\n"
        }
        return out
    }

    public static func links(_ concept: Concept) -> String {
        var out = "\(concept.id)\n  out: "
        out += concept.links.isEmpty
            ? "-" : concept.links.joined(separator: " ")
        out += "\n  in:  "
        out += concept.backlinks.isEmpty
            ? "-" : concept.backlinks.joined(separator: " ")
        out += "\n"
        return out
    }

    public static func advice(_ store: Store, _ id: String) -> String {
        var out = ""
        if let report = store.report(for: id) {
            if !report.strayArea.isEmpty {
                out += "note: its content sits nearest "
                out += report.strayArea + "/, not "
                out += Store.area(of: id) + "/\n"
            }
            if !report.candidates.isEmpty {
                out += "consider linking: "
                out += report.candidates.joined(separator: " ") + "\n"
            }
        }
        return out
    }

    public static func saved(_ store: Store, id: String, existed: Bool) -> String {
        var out = "\(existed ? "updated" : "created") \(id)"
            + "  (\(store.concepts.count) concepts,"
            + " re-embedded \(store.embeddedCount))\n"
        out += advice(store, id)
        return out
    }

    public static func retired(_ store: Store, id: String,
                        referrers: [String]) -> String {
        let live = store.concepts.filter { concept in
            !concept.isDeprecated
        }.count
        var out = "retired \(id)  (\(live) live concepts)\n"
        if !referrers.isEmpty {
            out += "still linking to it, and still resolving: "
            out += referrers.joined(separator: " ") + "\n"
        }
        return out
    }

    public static func purged(_ store: Store, id: String,
                       referrers: [String]) -> String {
        var out = "purged \(id)  (\(store.concepts.count) concepts"
        out += " remain)\n"
        if !referrers.isEmpty {
            out += "still linking to it, now dangling: "
            out += referrers.joined(separator: " ") + "\n"
        }
        return out
    }

    public static func advice(_ kind: Proposal.Kind) -> String {
        var out = ""
        switch kind {
        case .dangling:
            out = "the first links to the second, which does not exist: "
                + "okf_update the first so the sentence still reads"
        case .merge:
            out = "near-duplicates: read both, fold into one with "
                + "okf_update, okf_forget the other"
        case .link:
            out = "related but unlinked: okf_update one to link the other"
        case .skill:
            out = "these cluster with nothing naming the whole: "
                + "okf_create one concept that says what they share and "
                + "links each, if that concept is real"
        case .orphan:
            out = "unconnected: okf_update it to link what it belongs with"
        }
        return out
    }

    public static func dream(_ store: Store, _ proposals: [Proposal]) -> String {
        var out = proposals.isEmpty
            ? "nothing to consolidate\n"
            : "\(proposals.count) proposals\n\n"
        for proposal in proposals {
            out += String(format: "%@  %.2f\n", proposal.kind.rawValue,
                          proposal.score)
            for id in proposal.members {
                let title = store.concept(id)?.title ?? id
                out += "  \(id)  \(title)\n"
            }
            out += "  -> " + advice(proposal.kind) + "\n\n"
        }
        return out
    }

    struct Summary {
        var types: [String: Int] = [:]
        var bodyBytes = 0
        var abstractBytes = 0
        var edges = 0
        var stale = 0
        var orphans = 0
        var dangling = 0
    }

    static func summarize(_ store: Store) -> Summary {
        var out = Summary()
        for concept in store.concepts {
            out.types[concept.type, default: 0] += 1
            out.bodyBytes += concept.body.utf8.count
            out.abstractBytes += concept.passage.utf8.count
            out.edges += concept.links.count
            if concept.isStale { out.stale += 1 }
            if concept.links.isEmpty && concept.backlinks.isEmpty {
                out.orphans += 1
            }
            for link in concept.links {
                if store.concept(link) == nil { out.dangling += 1 }
            }
        }
        return out
    }

    public static func stats(_ store: Store, _ embedder: Embedder) -> String {
        let summary = summarize(store)
        let count = max(1, store.concepts.count)
        let ranked = summary.types.sorted { left, right in
            (-left.value, left.key) < (-right.value, right.key)
        }
        var out = "concepts   \(store.concepts.count)\n"
        out += "edges      \(summary.edges) out, avg "
        out += String(format: "%.1f",
                      Double(summary.edges) / Double(count))
        out += " per concept, \(summary.dangling) dangling\n"
        out += "orphans    \(summary.orphans)\n"
        out += "stale      \(summary.stale)\n"
        out += "body       \(summary.bodyBytes)B total, "
        out += "\(summary.bodyBytes / count)B avg, "
        out += "~\(estimatedTokens(bytes: summary.bodyBytes)) tok\n"
        out += "abstracts  \(summary.abstractBytes)B total, "
        out += "\(summary.abstractBytes / count)B avg, "
        out += "~\(estimatedTokens(bytes: summary.abstractBytes)) tok\n"
        out += "vectors    \(count) x \(embedder.dim) f32 x 2 = "
        out += "\(count * embedder.dim * 8)B (\(embedder.name))\n"
        out += "types      "
        out += ranked.map { entry in "\(entry.key) \(entry.value)" }
            .joined(separator: ", ") + "\n"
        return out
    }

    public static func similarity(_ embedder: Embedder, _ left: String,
                           _ right: String) -> String {
        let query = embedder.embedQuery(left)
        let passage = embedder.embedPassage(right)
        var total: Float = 0
        for i in 0..<min(query.count, passage.count) {
            total += query[i] * passage[i]
        }
        return String(format: "%.4f  %@ | %@", total, left, right)
    }

    public static func tokens(_ embedder: Embedder, _ text: String) -> String {
        embedder.tokens(text).map { id in String(id) }
            .joined(separator: " ")
    }
}
