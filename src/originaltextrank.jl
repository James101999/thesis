using WordTokenizers
using TextAnalysis
using TextModels
using Graphs
using PyCall
using CSV
using DataFrames

# Neural-based PoS Tagger consuming too much memory
pos = PoSTagger()

# Corpus and other CSV will go here soon

# Call python libraries here
#function lemmatization(data)
#    py"""
#    import spacy
#    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
#    
#    def lemmatize(x):
#        doc = nlp(x)
#        return " ".join([token.lemma_ for token in doc])
#    """
#    py"lemmatize"(data)
#end

function coreference(data)
    py"""
    # src code: from huggingface's neuralcoref
    import spacy
    nlp = spacy.load('en')

    import neuralcoref
    neuralcoref.add_to_pipe(nlp)

    def pass_data(x):
        doc = nlp(x)
        return doc._.coref_resolved
    """
    py"pass_data"(data)
end
    
function postagging(data)
    # Tokenize the text
    #tok = tokenize(text(sd))
    tok = split_sentences(data)
    # Tag with parts of speech
    #pos = PoSTagger()::TextModels.PoSModel
    #tags = pos(text(sd)::String)::Vector{String}
    tags = pos.(tok)::Vector{Vector{String}}
    # unfiltered = collect(zip(tokenizer::Vector{String}, tags::Vector{String}))
    zipped = [collect(zip(nltk_word_tokenize(tok[i])::Vector{String}, tags[i]::Vector{String})) for i in eachindex(tok)]
    reduce(vcat, zipped)
end

function documentpreparation(data)
    sd = StringDocument(data)
    # Let us first remove the unreadable characters
    remove_corrupt_utf8!(sd)
    # lower case every word
    remove_case!(sd)
    # Too aggressive, replace with lemmatization
    # stem!(sd)
    sd
end

function preprocessing(document)
    sd = documentpreparation(document)
     # stripping articles and stopwords
    prepare!(sd, strip_articles | strip_stopwords)
    tags = postagging(text(sd))
    # Filter tagged vocabulary
    filtered = filter(t -> ((_, y) = t;
        y == "NN" || y == "NNS" || y == "NNP" || y == "NNPS"
            || y == "JJ" || y == "JJR" || y == "JJS"), tags)
    Corpus([TokenDocument([x for (x, _) in filtered])])
end

function cooccurrencematrix(corpus)
    # Build a co-occurrence matrix of words
    CooMatrix(corpus, window = 2, normalize = false)
end

function buildgraph(cooccurrencematrix)
    # Build a graph from the sparse matrix of co-occurrences
    Graph(coom(cooccurrencematrix)::AbstractMatrix)
end

function score(data)
    corpus = preprocessing(data)
    coomatrix = cooccurrencematrix(corpus)
    graphmatrix = buildgraph(coomatrix)
    # The scoring algorithm
    # An efficient built in implementation from Graphs.jl
    # Saw another more efficient implementation that uses BLAS, will look into it soon
    score = pagerank(graphmatrix, 0.85, 30, 1.0e-4)::Vector{Float64} # this may cause an error on non-64 bit systems
    sort(collect(zip(score, coomatrix.terms)); rev = true)[1:floor(Int, 1/3 * length(coomatrix.terms))]
    # map(x -> x[2], sortedScore)[1:floor(Int, 1/3 * length(coomatrix.terms))]
end

function postprocessing(data)
    scores = score(data)
    document = documentpreparation(data)
    prepare!(document, strip_punctuation)
    lex = nltk_word_tokenize(text(document))
    toprankwords = [x[2] for x in scores]
    toprankscores = [x[1] for x in scores]
    newwords = []
    multiword = ""

    for i in eachindex(lex)
        if lex[i] ∈ toprankwords
            multiword *= lex[i] * " "
        else
            if occursin(multiword, text(document)) && !isempty(multiword)
                push!(newwords, rstrip(multiword))
            end
            multiword = ""
        end
    end

    phrases = collect(Set(newwords))
    final = []
    for i in eachindex(phrases)
        splittedphrase = split(phrases[i]::AbstractString)
        tempscore::AbstractFloat = 0.0
        if length(splittedphrase) < 2
            push!(final, (toprankscores[findall(x->x==phrases[i], 
            toprankwords)[1]::Integer], phrases[i]::AbstractString))
        else
            for j in eachindex(splittedphrase)
                tempscore += toprankscores[findall(x->x==splittedphrase[j], 
                toprankwords)[1]::Integer]
            end
            push!(final, (tempscore, phrases[i]::AbstractString))
            tempscore = 0
        end
    end
    map(x -> x[2], sort(final; rev=true))::Vector{SubString{String}}


    #joinwhitespace(x) = join(x, " ")
    #phrases = map(joinwhitespace, collect(Iterators.product(cooccurrencematrix.terms)))
    #occuringphrase(x) = occursin(x, text(processeddocument))
    #filtered = filter(occuringphrase, phrases)

end
