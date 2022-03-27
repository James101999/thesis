using WordTokenizers
using TextAnalysis
using TextModels
using Graphs

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

function preprocessing(string)
    # Document Preparation
    sd = StringDocument(string)
    # Stemming the text document and removing corrupted characters
    remove_corrupt_utf8!(sd)
    remove_case!(sd)
    prepare!(sd, strip_articles | strip_stopwords)
    # Tokenize the text
    tokenizer = tokenize(text(sd))
    # Tag with parts of speech
    pos = PoSTagger()
    tags = pos(text(sd))
    # Filter tagged vocabulary
    unfiltered = collect(zip(tokenizer, tags))
    filtered = filter(t -> ((_, y) = t;
        y == "NN" || y == "NNS" || y == "NNP" || y == "NNPS"
            || y == "VB"), unfiltered)
    Corpus([StringDocument(join([x for (x, _) in filtered], " "))])
end

function buildgraph(corpus)
    # Build a co-occurrence matrix of words
    cooccurrence = CooMatrix(corpus, window = 1, normalize = false)
    # Build a graph from the sparse matrix of co-occurrences
    cooccurrence, squash(Graph(coom(cooccurrence)))
end

function textrank(data)
    corpus = preprocessing(data)
    coomatrix, graphmatrix = buildgraph(corpus)
    # The scoring algorithm
    # An efficient built in implementation from Graphs.jl
    score = pagerank(graphmatrix)
    if length(coomatrix.terms) < 10
        collect(values(sort(Dict(score .=> coomatrix.terms); rev = true)))
    else
        collect(values(sort(Dict(score .=> coomatrix.terms); rev = true)))[1:10]
    end
end
