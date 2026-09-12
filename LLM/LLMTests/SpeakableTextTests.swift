import Foundation
import Testing
@testable import LLM

// What a voice should say about a Markdown answer, and -- as much -- what it
// should NOT say.

// Everything a chunker emits for one whole input, pushed in one go.
private func segments(_ text: String) -> [Segment] {
    var s = SpeakableText()
    var out = s.push(text)
    out.append(contentsOf: s.finish())
    return out
}

private func spoken(_ text: String) -> [String] {
    segments(text).map { s in s.spoken }
}

private func shown(_ text: String) -> [String] {
    segments(text).map { s in s.shown }
}

// The same input delivered one character at a time, which is closer to how a
// token stream actually arrives.
private func spokenByCharacter(_ text: String) -> [String] {
    var s = SpeakableText()
    var out: [Segment] = []
    for ch in text { out.append(contentsOf: s.push(String(ch))) }
    out.append(contentsOf: s.finish())
    return out.map { s in s.spoken }
}

@Suite struct SpeakableTextTests {

    @Test func sentencesComeOutOneAtATime() {
        let out = spoken("One thing. Two things! Three?\n")
        #expect(out == ["One thing.", "Two things!", "Three?"])
    }

    // The whole point of streaming: sentence one is speakable before the rest
    // of the paragraph has been generated.
    @Test func aCompleteSentenceIsEmittedBeforeItsLineEnds() {
        var s = SpeakableText()
        let first = s.push("Ready to go. And then")
        #expect(first.map { p in p.spoken } == ["Ready to go."])
    }

    // Chunk boundaries are an artifact of the token stream and must not be able
    // to change what is said.
    @Test func chunkingDoesNotChangeTheResult() {
        let text = "A **first** line. A second one!\n\nAnd 1999?\n"
        #expect(spoken(text) == spokenByCharacter(text))
    }

    // A sentence that ends before its line does is the COMMON case while
    // streaming, and it has to be shaped like any other.
    @Test func aMidLineSentenceIsStillShaped() {
        var s = SpeakableText()
        let out = s.push("It cost 2,925.26 in **1999**. And then")
        #expect(out.map { p in p.spoken }
                == ["It cost two thousand nine hundred twenty-five "
                    + "point two six in nineteen ninety-nine."])
    }

    // Marker stripping belongs to the START of a line only; a hash or a
    // "1." arriving mid-sentence is text.
    @Test func markersAreNotStrippedMidLine() {
        var s = SpeakableText()
        _ = s.push("First one. ")
        let out = s.push("Tagged #top and - dashed. ")
        #expect(out.map { p in p.spoken }
                == ["Tagged #top and - dashed."])
    }

    // "1,000" reached the phonemizer whole only when its sentence ended at a
    @Test func aGroupedThousandIsNotReadAsOneZero() {
        var s = SpeakableText()
        let out = s.push("Starting with 1,000 units. ")
        #expect(out.map { p in p.spoken }
                == ["Starting with one thousand units."])
    }

    @Test func aDecimalPointAtAChunkBoundaryIsNotASentenceEnd() {
        let text = "So, you need 227.5 grams of butter.\n"
        #expect(spokenByCharacter(text)
                == ["So, you need two hundred twenty-seven point five "
                    + "grams of butter."])
        #expect(spokenByCharacter(text) == spoken(text))
    }

    @Test func aGroupedThousandSurvivesAChunkBoundary() {
        let text = "The total is 1,558 tokens.\n"
        #expect(spokenByCharacter(text)
                == ["The total is one thousand five hundred fifty-eight "
                    + "tokens."])
        #expect(spokenByCharacter(text) == spoken(text))
    }

    @Test func aDecimalIsWholeHoweverItIsChunked() {
        let text = "Pi is 3.14 exactly.\n"
        #expect(spokenByCharacter(text) == spoken(text))
    }

    @Test func displayMathIsDescribedNotRead() {
        let out = spoken("""
        To find out how many grams of butter you need:

        $$
        \\text{Butter weight} = \\text{Flour weight} \\times 0.65
        $$

        So, you need 227.5 grams of butter.
        """)
        #expect(out == ["To find out how many grams of butter you need:",
                        "An equation.",
                        "So, you need two hundred twenty-seven point five "
                            + "grams of butter."])
    }

    @Test func aOneLineDisplayIsDescribedToo() {
        let out = spoken("Before.\n\n$$ x = 1 $$\n\nAfter.\n")
        #expect(out == ["Before.", "An equation.", "After."])
    }

    @Test func anUnclosedDisplayIsStillDescribed() {
        let out = spoken("Look:\n\n$$\n\\frac{a}{b}\n")
        #expect(out == ["Look:", "An equation."])
    }

    @Test func displayMathSurvivesCharacterChunking() {
        let text = "Given:\n\n$$\n\\text{a} \\times 2\n$$\n\nDone.\n"
        #expect(spokenByCharacter(text) == spoken(text))
        #expect(spoken(text) == ["Given:", "An equation.", "Done."])
    }

    @Test func codeBlocksAreDescribedNotRead() {
        let out = spoken("""
        Here it is:

        ```python
        for i in range(10):
            print(i)
        ```

        That was the loop.
        """)
        #expect(out == ["Here it is:", "A python code block.",
                        "That was the loop."])
    }

    @Test func anUnlabelledCodeBlockStillGetsDescribed() {
        let out = spoken("```\nx = 1\n```\n")
        #expect(out == ["A code block."])
    }

    // An unterminated fence at end of turn (a stopped generation) must still
    // close, or the rest of the answer is swallowed.
    @Test func anUnclosedFenceIsStillDescribed() {
        let out = spoken("Look:\n\n```swift\nlet x = 1\n")
        #expect(out == ["Look:", "A swift code block."])
    }

    // A table read cell by cell loses the geometry that made it a table, so
    // the count is the useful part. The header and its rule are not rows.
    @Test func tablesAreCountedNotRead() {
        let out = spoken("""
        Results:

        | Model | Size |
        |---|---|
        | A | 1 |
        | B | 2 |
        | C | 3 |

        Done.
        """)
        #expect(out == ["Results:", "A table of three rows.", "Done."])
    }

    @Test func listsKeepTheirItemsAndLoseTheirBullets() {
        let out = spoken("- first item\n- second item\n3. third item\n")
        #expect(out == ["first item", "second item", "third item"])
    }

    @Test func aNumberedMarkerIsNotSpokenAsItsOwnSentence() {
        let text = "1. First point here.\n2. Second point here.\n"
        #expect(spoken(text) == ["First point here.",
                                 "Second point here."])
        #expect(spokenByCharacter(text) == spoken(text))
    }

    @Test func headingsAreTheirOwnBreath() {
        let out = spoken("## Overview\nThe body follows.\n")
        #expect(out == ["Overview", "The body follows."])
    }

    @Test func emphasisAndBackticksAreNotSpoken() {
        let out = spoken("It is **bold**, _italic_ and `code`.\n")
        #expect(out == ["It is bold, italic and code."])
    }

    @Test func aLinkKeepsItsTextAndDropsItsTarget() {
        let out = spoken("See [the docs](https://example.com/a) for more.\n")
        #expect(out == ["See the docs for more."])
    }

    // The three ways a full stop lies about ending a sentence. The decimal
    // arrives as words, because numbers are expanded on the way through.
    @Test func abbreviationsDoNotEndSentences() {
        #expect(spoken("Dr. Smith arrived.\n") == ["Dr. Smith arrived."])
        #expect(spoken("Pi is 3.14 exactly.\n")
                == ["Pi is three point one four exactly."])
        #expect(spoken("J. R. R. Tolkien wrote it.\n")
                == ["J. R. R. Tolkien wrote it."])
    }

    @Test func aHorizontalRuleIsSilent() {
        let out = spoken("Before.\n\n---\n\nAfter.\n")
        #expect(out == ["Before.", "After."])
    }

    // A turn that ends without punctuation (a stopped generation) must not
    // lose its tail.
    @Test func anUnterminatedTailIsStillSpoken() {
        let out = spoken("This one just trails off")
        #expect(out == ["This one just trails off"])
    }

    @Test func emptyInputSaysNothing() {
        #expect(spoken("").isEmpty)
        #expect(spoken("\n\n   \n").isEmpty)
    }

    @Test func theShownFormKeepsWhatIsOnScreen() {
        let src = "A high of 99 degrees today. **Wednesday:** 101 degrees.\n"
        #expect(shown(src) == ["A high of 99 degrees today.",
                               "Wednesday: 101 degrees."])
        #expect(spoken(src) == ["A high of ninety-nine degrees today.",
                                "Wednesday: one hundred one degrees."])
    }
}
