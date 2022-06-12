# Morphological guesser with lexd How-To

## Disclaimer

This instruction summarizes my personal experience and does not necessarily show the optimal way to achieve the desired result. 

***

## Requirements

This instruction assumes that you have already implemented a morphological analyser based on lexd.

In order to produce a finite state guesser, you need to search for all the distinct paradigms that your language has, and to create a tag for each one of them in a separate lexicon. This means that a complete grammar should be accessible.

***

## Summary

In short, creating a morhological guesser includes the following steps:

 - Create paradigm tags and add them to your main lexd file
 - Make a normal morphological analyser using the new lexd file
 - Substitute paradigm tags with special tags \[GUESS_CATEGORY=...\] using hfst-substitute
 - Pass the analyser with substitutions to hfst-guessify
 - Enjoy your guesser

Each of those steps will be characterized in detail in further sections.

***

## Implementation

### Folder

Firstly, you can create a separate folder inside your project to hold all the guesser-related files. This is useful, if you don't want the many intermediate files, that should be produced, to interfere with your base project.

***

### Add a paradigm tag lexicon

Assuming that the language only has a verbal and a nominal paradigm, the lexicon with paradigm tags would look like this:

```
LEXICON DeclType
<TypeVerb>:[v]
<TypeNoun>:[n]
```

Note that those tags are not always equal to part of speech tags, as you can have different paradigms for one and the same POS. If some of the words in your language cannot be declined (adverbs, clitics, etc.), you don't need a paradigm tag for them, so just ignore them.

We presume, that paradigm tags are not externalized on the surface side, e.g. the right side after the colon should be empty. As for tag names, you are free to choose any, but they must have an equal prefix or suffix, so that these tags can be selected by **grep**.

The lexicon can be stored in a separate lexd file or in your main lexd file, if you don't mind having additional tags in your analyses. 

***

### Make a modified lexd file

If you have created tags for all paradigms, the next step involves modification of your main lexd file. If you don't want to change it, you can create a copy to operate on inside the guesser folder.

In the file you work with, you should modify the existing rules (**PATTERNS** section) and insert paradigm tags **strictly after** the corresponding roots. The difference is illustrated by the following code snippets.

Regular lexd file:
```
PATTERNS
NounRoot NounDecl
VerbRoot VerbDecl

LEXICON VerbDecl
<3SG><PRES>:s
<INF>:

LEXICON NounDecl
<SG>:
<PL>:s

LEXICON NounRoot
dog<N>:dog
cat<N>:cat

LEXICON VerbRoot
beat<V>:beat
writhe<V>:writhe
```

Modified lexd file:
```
PATTERNS
NounRoot DeclType[n] NounDecl
VerbRoot DeclType[v] VerbDecl

LEXICON DeclType
<TypeVerb>:[v]
<TypeNoun>:[n]

LEXICON VerbDecl
<3SG><PRES>:s
<INF>:

LEXICON NounDecl
<SG>:
<PL>:s

LEXICON NounRoot
dog<N>:dog
cat<N>:cat

LEXICON VerbRoot
beat<V>:beat
writhe<V>:writhe
```

Note that DeclType lexicon was added and that DeclType items were inserted after roots in the second snippet.

If you already have a large lexd file with lots of patterns that are hard to change by hand, you can rely on sed to insert DeclType items. Of course, this requires you to remember names of lexicons which hold roots that belong to different paradigms.

Example: here, we substitute "NounRoot " with "NounRoot Decltype\[n\] "
```
sed -i 's/^\(NounRoot \)/\1DeclType[n] /g' target.lexd
sed -i 's/^\(VerbRoot \)/\1DeclType[v] /g' target.lexd
```

***

### Compile an analyser

The next step is to follow the standard procedure of creating a morphological analyser, but with your brand new lexd file. Intersect it with a twol file, if you have one, and then invert.

```
lexd target.lexd | hfst-txt2fst | hfst-compose-intersect - target.twol.hfst | hfst-invert - -o target.ana.hfst
```

***

### Substitute paradigm tags with \[GUESS_CATEGORY=...\]

*hfst-guessify* utility that we are going to use requires special tags \[GUESS_CATEGORY=x\] to be present in the analysis. These are used to summarize surface forms inside each of the categories and to condition the guesser in this manner. So, we just need to substitute the paradigm tags inside the new analyser with those special tags.

This can be done with the following command:

```
hfst-summarize -v target.ana.hfst \
	| grep -A1 'sigma set:' \
	| tail -1 \
	| sed 's/, /\n/g' \
	| grep -E '<Type.+>'\
	| sed 's/<Type\(.\+\)>/&\t[GUESS_CATEGORY=\1]/'\
	| hfst-substitute -F - -o target.subst.hfst target.ana.hfst
```

Explanation: *hfst-substitute* needs a tsv file, in which substituted symbols are separated by tab from substituting symbols.
Firstly, we use *hfst-summarize* to access all the tags in the analyzer (mind the **-v** flag and **grep** and **tail** commands that come after). Then, we filter the paradigm tags with grep and form a tsv file with sed.

This intermediate file looks like this:
```
<TypeNoun>  [GUESS_CATEGORY=Noun]
<TypeVerb>  [GUESS_CATEGORY=Verb]
```

Then, we immediately pass it to hfst-substitute together with the analyzer, after which instead of
```
<N><TypeNoun><SG>
```
the resulting file *target.subst.hfst* should produce the following analysis string:
```
<N>[GUESS_CATEGORY=Noun]<SG>
```

You can test this new file on some of the surface forms to make sure, that it behaves as expected.

***

### Result filtering (optional)

Optionally, you can use XEROX-type regular expressions to filter out undesired results, like predictions of nominative case or predictions of absolutive case: *hfst-guess* utility is always tempted to assume that the surface form you analyze is just a nominative form of a noun and has no suffixes whatsoever.

For this to work, create a file with a regular expression, compile it with *hfst-regexp2fst*, and compose your guesser with it.

This expression allows all symbols but the absolutive tag. If you use it, the guesser will not predict the absolutive case.
```
[ ? - "<abs>" ]*
```
Here, we create a transducer from the regexp and compose the analyser with it.
```
hfst-regexp2fst -o filter.hfst < filter.regexp
hfst-compose -F -1 target.subst.hfst -2 filter.hfst | hfst-minimize > target.filtered.hfst
```

***

### Guessifying

If you have an analyser with paradigm tags substituted by \[GUESS_CATEGORY=...\], you are ready to pass it to *hfst-guessify* utility and produce a guesser.

The command is as simple, as:
```
hfst-guessify -v  target.filtered.hfst > target.guesser.hfst
```
or, if you haven't applied a filter:
```
hfst-guessify -v  target.subst.hfst > target.guesser.hfst
```

***

## How to guess

When this is done, you should be able to analyze any surface form using *hfst-guess* and the following command:
```
echo "yourword" | hfst-guess target.guesser.hfst
```

***

## Problems

### Why not POS-tags?

You might ask, why not to try substituting POS-tags with guess category tags instead of inserting some new paradigm tags? In fact, I've tried it and it did not work at least in my case.

***

### Clitics

If your language has some postfixes that can be added to a word from any paradigm, like postpositions, then you are better off removing them from your modified lexd file. They will probably be over-represented across all paradigms, so *hfst-guess* will be eager to find them in each and every analysis.

***

### Prefixes

The guesser that I have made is ignoring prefixes, like the agreement markers, since they come *before* the root. This results from the fact that *hfst-guessify* only considers parts of words that come *after* the \[GUESS_CATEGORY=x\] symbol. This problem can probably be solved by means of a dirty hack, if you insert a guess category after the prefixes (initial agreement markers in my case) in your **PATTERNS** and make the guesser believe that forms that include those prefixes are compounds. Personally, I haven't tried this just yet. 