#!/usr/bin/env python3
import networkx as nx
import nltk
import spacy
import pytextrank
# Build the Graph

# spacy pytextrank
# example text
#text = "Compatibility of systems of linear constraints over the set of natural numbers. Criteria of compatibility of a system of linear Diophantine equations, strict inequations, and nonstrict inequations are considered. Upper bounds for components of a minimal set of solutions and algorithms of construction of minimal generating sets of solutions for all types of systems are given. These criteria and the corresponding algorithms for constructing a minimal supporting set of solutions can be used in solving all the considered types systems and systems of mixed types."
text2 = "I like Python because I can build AI applications"
#text = "I like Python because I can do data analytics"
token = nltk.word_tokenize(text)
nltk.pos_tag(token)

# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")

# add PyTextRank to the spaCy pipeline
# nlp.add_pipe("textrank")
doc = nlp(text)
text = "(1st UPDATE) 'The events in Ukraine are truly tragic. We urgently need peace,' says billionaire Andrey Melnichenko, who is Russian but was born in Belarus and has a Ukrainian mother
LONDON, United Kingdom – A global food crisis looms unless the war in Ukraine is stopped because fertilizer prices are soaring so fast that many farmers can no longer afford soil nutrients, Russian fertilizer and coal billionaire Andrey Melnichenko said on Monday, March 14.

Several of Russia’s richest businessmen have publicly called for peace since President Vladimir Putin ordered the invasion on February 24, including Mikhail Fridman, Pyotr Aven, and Oleg Deripaska.

The United States and its European allies have cast Putin’s invasion as an imperial-style land grab that has so far been poorly executed because Moscow underestimated Ukrainian resistance and Western resolve to punish Russia.

The West has sanctioned Russian businessmen, including European Union sanctions on Melnichenko, frozen state assets, and cut off much of the Russian corporate sector from the global economy in an attempt to force Putin to change course.

Putin refuses to. He has called the war a special military operation to rid Ukraine of dangerous nationalists and Nazis.

“The events in Ukraine are truly tragic. We urgently need peace,” Melnichenko, 50, who is Russian but was born in Belarus and has a Ukrainian mother, told Reuters in a statement emailed by his spokesman.

“One of the victims of this crisis will be agriculture and food,” said Melnichenko, who founded EuroChem, one of Russia’s biggest fertilizer producers, which moved to Zug, Switzerland, in 2015, and SUEK, Russia’s top coal producer.

Russia’s invasion of Ukraine has killed thousands, displaced more than 2 million people, and raised fears of a wider confrontation between Russia and the United States, the world’s two biggest nuclear powers.

Food war?
Putin warned last Thursday, March 10, that food prices would rise globally due to soaring fertilizer prices if the West created problems for Russia’s export of fertilizers – which account for 13% of world output.

Russia is a major producer of potash, phosphate, and nitrogen containing fertilizers – major crop and soil nutrients. EuroChem, which produces nitrogen, phosphates, and potash, says it is one of the world’s top five fertilizer companies.

The war “has already led to soaring prices in fertilizers which are no longer affordable to farmers,” Melnichenko said.

He said food supply chains already disrupted by COVID-19 were now even more distressed.

“Now it will lead to even higher food inflation in Europe and likely food shortages in the world’s poorest countries,” he said.

Russia’s trade and industry ministry told the country’s fertilizer producers to temporarily halt exports earlier this month."
# examine the top-ranked phrases in the document
for phrase in doc._.phrases:
    print(phrase.text)
    print(phrase.rank, phrase.count)
    print(phrase.chunks)
