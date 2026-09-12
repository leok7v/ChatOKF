import Foundation
import Testing
@testable import LLM

// Numerals as a voice should say them.
@Suite struct SpokenNumbersTests {

    @Test func plainIntegersAreCardinals() {
        #expect(SpokenNumbers.say("0") == "zero")
        #expect(SpokenNumbers.say("7") == "seven")
        #expect(SpokenNumbers.say("42") == "forty-two")
        #expect(SpokenNumbers.say("100") == "one hundred")
        #expect(SpokenNumbers.say("305") == "three hundred five")
    }

    // The reported case, end to end.
    @Test func aGroupedDecimalReadsAsOneNumber() {
        #expect(SpokenNumbers.say("2,925.26")
                == "two thousand nine hundred twenty-five point two six")
    }

    // Both separators present: the LAST one is the decimal, whichever it is.
    // The same quantity written either way must read the same.
    @Test func theLastSeparatorIsTheDecimal() {
        let us = SpokenNumbers.say("1,234,567.89")
        let eu = SpokenNumbers.say("1.234.567,89")
        #expect(us == eu)
        #expect(us == "one million two hundred thirty-four thousand "
                + "five hundred sixty-seven point eight nine")
    }

    // One separator, so the digit count after it decides.
    @Test func aLoneSeparatorIsReadByItsGrouping() {
        #expect(SpokenNumbers.say("1,234") == "one thousand two hundred "
                + "thirty-four")
        #expect(SpokenNumbers.say("1,5") == "one point five")
        #expect(SpokenNumbers.say("3.14") == "three point one four")
        #expect(SpokenNumbers.say("0.75") == "zero point seven five")
    }

    // A fraction is spoken digit by digit, never as a quantity: "point two
    // six", not "point twenty-six".
    @Test func fractionsAreSpelledDigitByDigit() {
        #expect(SpokenNumbers.say("0.26") == "zero point two six")
        #expect(SpokenNumbers.say("1.05") == "one point zero five")
    }

    @Test func yearsReadAsYears() {
        #expect(SpokenNumbers.year(1999) == "nineteen ninety-nine")
        #expect(SpokenNumbers.year(2026) == "twenty twenty-six")
        #expect(SpokenNumbers.year(1900) == "nineteen hundred")
        #expect(SpokenNumbers.year(1100) == "eleven hundred")
        #expect(SpokenNumbers.year(1905) == "nineteen oh five")
    }

    // 2005 is the one a listener expects as a quantity rather than "twenty oh
    // five", and the band stops at 2099.
    @Test func theYearReadingHasLimits() {
        #expect(SpokenNumbers.say("2005") == "two thousand five")
        #expect(SpokenNumbers.say("1099") == "one thousand ninety-nine")
        #expect(SpokenNumbers.say("2100") == "two thousand one hundred")
        // A grouped four-digit number is a quantity, never a year.
        #expect(SpokenNumbers.say("1,999") == "one thousand nine hundred "
                + "ninety-nine")
    }

    // Leading zeros mark an identifier, not a quantity.
    @Test func leadingZerosAreSpelledOut() {
        #expect(SpokenNumbers.say("007") == "zero zero seven")
    }

    @Test func signAndPercentAreSpoken() {
        #expect(SpokenNumbers.expand("-5 degrees") == "minus five degrees")
        #expect(SpokenNumbers.expand("50% done") == "fifty percent done")
    }

    // A digit touching letters is part of a word, and the rules engine
    // already knows what to do with it.
    @Test func digitsInsideWordsAreLeftAlone() {
        #expect(SpokenNumbers.expand("H2O") == "H2O")
        #expect(SpokenNumbers.expand("the 26th") == "the 26th")
        #expect(SpokenNumbers.expand("3D model") == "3D model")
    }

    @Test func trailingPunctuationStaysOutsideTheNumber() {
        #expect(SpokenNumbers.expand("It cost 25.")  == "It cost twenty-five.")
        #expect(SpokenNumbers.expand("(see 3), yes")
                == "(see three), yes")
    }

    @Test func anAmountReadsAsMoney() {
        #expect(SpokenNumbers.expand("It costs $2,925.26.")
                == "It costs two thousand nine hundred twenty-five dollars "
                + "and twenty-six cents.")
    }

    // The unit agrees with the count, and a whole amount says no cents at
    // all rather than "and zero cents".
    @Test func theUnitIsPluralUnlessItIsOne() {
        #expect(SpokenNumbers.expand("$5") == "five dollars")
        #expect(SpokenNumbers.expand("$1") == "one dollar")
        #expect(SpokenNumbers.expand("$1.00") == "one dollar")
        #expect(SpokenNumbers.expand("$3.01")
                == "three dollars and one cent")
    }

    // Under a whole unit, naming the majors is noise.
    @Test func aSubUnitAmountDropsTheMajor() {
        #expect(SpokenNumbers.expand("$0.26") == "twenty-six cents")
        #expect(SpokenNumbers.expand("50\u{00A2}") == "fifty cents")
        #expect(SpokenNumbers.expand("1\u{00A2}") == "one cent")
    }

    @Test func eachCurrencyKeepsItsOwnSubUnit() {
        #expect(SpokenNumbers.expand("£3.50")
                == "three pounds and fifty pence")
        #expect(SpokenNumbers.expand("€7.05")
                == "seven euros and five cents")
        #expect(SpokenNumbers.expand("₹1,200.75")
                == "one thousand two hundred rupees and seventy-five paise")
        // Yen has no subunit, so two digits stay a decimal quantity.
        #expect(SpokenNumbers.expand("¥500") == "five hundred yen")
    }

    // Exactly two fraction digits are subunits. One digit is a quantity, and
    // inventing "fifty cents" from it would be a guess.
    @Test func onlyTwoFractionDigitsAreSubUnits() {
        #expect(SpokenNumbers.expand("$1.5") == "one point five dollars")
    }

    // Past a million the figure is usually written with the scale as a word,
    // and the unit belongs after it.
    @Test func aScaleWordTakesTheUnitAfterIt() {
        #expect(SpokenNumbers.expand("$1.5 million")
                == "one point five million dollars")
        #expect(SpokenNumbers.expand("$3 billion") == "three billion dollars")
        // No symbol, so nothing is appended.
        #expect(SpokenNumbers.expand("1.5 million") == "one point five million")
    }

    // A forecast is mostly trailing signs, and every one of them is silent
    // unless it is spelled out.
    @Test func temperaturesReadAsDegrees() {
        #expect(SpokenNumbers.expand("61\u{00B0}F Clear")
                == "sixty-one degrees Fahrenheit Clear")
        #expect(SpokenNumbers.expand("21\u{00B0}C")
                == "twenty-one degrees Celsius")
        #expect(SpokenNumbers.expand("a 90\u{00B0} turn")
                == "a ninety degrees turn")
        #expect(SpokenNumbers.expand("1\u{00B0}C") == "one degree Celsius")
    }

    @Test func aNegativeAmountSaysSo() {
        #expect(SpokenNumbers.expand("-$42.10")
                == "minus forty-two dollars and ten cents")
    }

    @Test func aVeryLargeNumberStillReads() {
        #expect(SpokenNumbers.say("12345678") == "twelve million three "
                + "hundred forty-five thousand six hundred seventy-eight")
    }
}

// The inline ordered list -- numbers that are scaffolding rather than
// quantities, and that also read as full stops.
@Suite struct InlineEnumeratorTests {

    @Test func aRunOfEnumeratorsIsDropped() {
        let out = SpeakableText.dropEnumerators(
            "Split into: 1. Definitions, 2. Differences, and 3. Evidence.")
        #expect(out == "Split into: Definitions, Differences, and Evidence.")
    }

    // One "N. " is far more likely to be a sentence that ended in a number,
    // so a lone match is left alone.
    @Test func aLoneNumberIsNotAnEnumerator() {
        let text = "It cost 25. Then we left."
        #expect(SpeakableText.dropEnumerators(text) == text)
    }

    @Test func aDecimalIsNeverAnEnumerator() {
        let text = "Values 1.5 and 2.5 and 3.5 differ."
        #expect(SpeakableText.dropEnumerators(text) == text)
    }
}
