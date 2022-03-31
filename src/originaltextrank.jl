using WordTokenizers
using TextAnalysis
using TextModels
using Languages
using Graphs
using PyCall

# Neural-based PoS Tagger consuming too much memory
pos = PoSTagger()

# This is just temporary, will move this test data soon
function data()
    """Statistical expert / data scientist / analytical developer
    BNOSAC (Belgium Network of Open Source Analytical Consultants), is a Belgium consultancy company specialized in data analysis and statistical consultancy using open source tools.
    In order to increase and enhance the services provided to our clients, we are on the lookout for an all-round statistical expert, data scientist and analytical developer.
    Function:
    Your main task will be the execution of a diverse range of consultancy services in the field of statistics and data science.
    You will be involved in a small team where you handle the consultancy services from the start of the project until the end.
    This covers:
    Joint meeting with clients on the topic of the analysis.
    Acquaintance with the data.
    Analysis of the techniques that are required to execute the study.
    Mostly standard statistical and biostatistical modelling, predictive analytics & machine learning techniques.
    Perform statistical design, modeling and analysis, together with more seniors.
    Building the report on the data analysis.
    Automating and R/Python package development.
    Integration of the models into the existing architecture.
    Giving advise to the client on the research questions, design or integration.
    Next to that, you will help in building data products and help sell them.
    These cover text mining, integration of predictive analytics in existing tools and the creation of specific data analysis tools and web services.
    You also might be involved in providing data science related courses for clients.
    Profile:
    You have a master degree in the domain of Statistics, Biostatistics, Mathematics, Commercial or Industrial Engineering, Economics or similar.
    You have a strong interest in statistics and data analysis.
    You have good communication skills, are fluent in English and know either Dutch or French.
    You soak up new knowledge and either just make things work or have the attitude of 'I can do this'.
    Besides this, you have attention to detail and adapt to changes quickly.
    You have programming experience in R or you really want to switch to using R.
    You have a sound knowledge of another data analysis language (Python, SQL, javascript) and you don't care in which relational database, Excel, bigdata or noSQL store your data is located.
    Interested in robotics is a plus.
    Offer:
    A half or full-time employment depending on your personal situation.
    The ability to get involved in a whole range of sectors and topics and the flexibility to shape your own future.
    The usage of a diverse range of statistical & data science techniques.
    Support in getting up to speed quickly in the usage of R.
    An environment in which you can develop your talent and make your own proposals the standard way to go.
    Liberty in managing your open source projects during working hours.
    Contact:
    To apply or in order to get more information about the job content, please contact us at: http://bnosac.be/index.php/contact/get-in-touch"""
end

# Corpus and other CSV will go here soon

function lemmatization(data)
    pyimport("spacy")
    whose(spacy)
    #nlp = spacy.load("en_core_web_sm")
    #lemmatizer = nlp.get_pipe("lemmatizer")
    #doc = nlp(data)
    #[token.lemma_ for token in doc].join(' ')
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
    CooMatrix(corpus, window = 3, normalize = false)
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
        splittedphrase = split(phrases[i]::String)
        tempscore::Integer = 0
        if length(splittedphrase) < 2
            push!(final, (toprankscores[findall(x->x==phrases[i], 
            toprankwords)[1]::Integer], phrases[i]::String))
        else
            for j in eachindex(splittedphrase)
                tempscore += toprankscores[findall(x->x==splittedphrase[j], 
                toprankwords)[1]::Integer]
            end
            push!(final, (tempscore, phrases[i]::String))
            tempscore = 0
        end
    end
    map(x -> x[2], sort(final; rev=true))::Vector{String}


    #joinwhitespace(x) = join(x, " ")
    #phrases = map(joinwhitespace, collect(Iterators.product(cooccurrencematrix.terms)))
    #occuringphrase(x) = occursin(x, text(processeddocument))
    #filtered = filter(occuringphrase, phrases)

end
