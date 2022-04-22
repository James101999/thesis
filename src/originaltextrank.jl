using WordTokenizers
using TextAnalysis
using TextModels
using Graphs
using PyCall
using CSV
using DataFrames
using BenchmarkTools
import Pandas

@pyimport nltk
@pyimport spacy
@pyimport neuralcoref
@pyimport datasets

# Neural-based PoS Tagger consuming too much memory
# Replaced with nltk pos tagger
# pos = PoSTagger()

# Corpus and other CSV will go here soon

function coreference(data::String)
    # src code: from huggingface's neuralcoref
    nlp = pycall(spacy.load, PyObject, PyObject("en"))
    pycall(neuralcoref.add_to_pipe, PyObject, nlp)
    doc = nlp(PyObject(data))
    doc._.coref_resolved
end

function coreference(data::Vector)
    # src code: from huggingface's neuralcoref
    nlp = pycall(spacy.load, PyObject, PyObject("en"))
    pycall(neuralcoref.add_to_pipe, PyObject, nlp)
    [x._.coref_resolved for x in nlp.pipe(data, batch_size=3000)]
end
 
#function postagging(data)
    # Tokenize the text
    #tok = nltk_word_tokenize(data)
    #tok = [x[1:end-1] for x in split_sentences(data)]
    # Tag with parts of speech
    #pos = PoSTagger()::TextModels.PoSModel
    #tags = pos(data::String)::Vector{String}
    #tags = pos.(tok)::Vector{Vector{String}}
    #collect(zip(tok::Vector{String}, tags::Vector{String}))
    #zipped = [collect(zip(nltk_word_tokenize(tok[i])::Vector{String}, tags[i]::Vector{String})) for i in eachindex(tok)]
    #reduce(vcat, zipped)
#end

function documentpreparation(data::String)
    sd = StringDocument(data)
    # Let us first remove the unreadable characters
    remove_corrupt_utf8!(sd)
    # lower case every word
    #remove_case!(sd)
    # Too aggressive, replace with lemmatization
    # stem!(sd)
    sd
end
    
function preprocessing(document::String)
    sd = documentpreparation(document)
    tok = nltk_word_tokenize(text(sd))
     # stripping articles and stopwords
    #prepare!(sd, strip_articles | strip_stopwords)
    pytok = PyVector(tok)
    tags = convert(Vector{Tuple{String, String}}, pycall(nltk.pos_tag, PyVector{Tuple{String, String}}, pytok))
    # Filter tagged vocabulary
    filtered = filter(t -> ((_, y) = t;
        y == "NN" || y == "NNS" || y == "NNP" || y == "NNPS"
            || y == "JJ" || y == "JJR" || y == "JJS"), tags)
    joined = TokenDocument([x for (x, _) in filtered])
    remove_case!(joined)
    Corpus([joined])
end

function preprocessing(document::Vector)
     # stripping articles and stopwords
    #prepare!(sd, strip_articles | strip_stopwords)
    pydoc = PyVector(document)
    tags = convert(Vector{Tuple{String, String}}, pycall(nltk.pos_tag, PyVector{Tuple{String, String}}, pydoc))
    # Filter tagged vocabulary
    filtered = filter(t -> ((_, y) = t;
        y == "NN" || y == "NNS" || y == "NNP" || y == "NNPS"
            || y == "JJ" || y == "JJR" || y == "JJS"), tags)
    joined = TokenDocument([x for (x, _) in filtered])
    remove_case!(joined)
    Corpus([joined])
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
    sort(collect(zip(score, coomatrix.terms)); rev = true)[1:ceil(Integer, 1 * length(coomatrix.terms))]
    #sort(collect(zip(score, coomatrix.terms)); rev = true)[1:ceil(Integer, 1/60 * length(coomatrix.terms))]
    # map(x -> x[2], sortedScore)[1:floor(Int, 1/3 * length(coomatrix.terms))]
end

function postprocessing(data::String)
    scores = score(data)
    document = documentpreparation(data)
    prepare!(document, strip_punctuation)
    remove_case!(document)
    lex = nltk_word_tokenize(text(document))
    toprankwords = [x[2] for x in scores]
    toprankscores = [x[1] for x in scores]
    newwords = Vector{SubString{String}}(undef, 1)
    multiword = ""

    for i in eachindex(lex)
        if lex[i] ∈ toprankwords
            multiword *= lex[i] * " "
        else
            if occursin(multiword, text(document)) && !isempty(multiword)
                if !isassigned(newwords)
                    newwords[1] = rstrip(multiword)
                else
                    push!(newwords, rstrip(multiword))
                end
            end
            multiword = ""
        end
    end

    phrases = collect(Set(newwords))
    final = Vector{Tuple{AbstractFloat, AbstractString}}(undef, length(phrases))
    splittedphrase = split.(phrases)
    for i in eachindex(phrases)
        tempscore::AbstractFloat = 0.0
        if length(splittedphrase[i]) < 2
            final[i] = (toprankscores[findall(x->x==phrases[i], 
            toprankwords)[1]::Integer], phrases[i]::AbstractString)
        else
            for j in eachindex(splittedphrase[i])
                tempscore += toprankscores[findall(x->x==splittedphrase[i][j], 
                toprankwords)[1]::Integer]
            end
            final[i] = (tempscore, phrases[i]::AbstractString)
            tempscore = 0
        end
    end
    map(x -> x[2], sort(final; rev=true))::Vector{SubString{String}}


    #joinwhitespace(x) = join(x, " ")
    #phrases = map(joinwhitespace, collect(Iterators.product(cooccurrencematrix.terms)))
    #occuringphrase(x) = occursin(x, text(processeddocument))
    #filtered = filter(occuringphrase, phrases)
end

function postprocessing(data::Vector)
    lex = TokenDocument(data)
    remove_case!(lex)
    lex_data = lex.tokens
    scores = score(data)
    toprankwords = [x[2] for x in scores]
    toprankscores = [x[1] for x in scores]
    newwords = Vector{SubString{String}}(undef, 1)
    multiword = ""
    for i in eachindex(lex_data)
        if lex_data[i] ∈ toprankwords
            multiword *= lex_data[i] * " "
        else
            if occursin(multiword, join(lex_data, ' ')) && !isempty(multiword)
                if !isassigned(newwords)
                    newwords[1] = rstrip(multiword)
                else
                    push!(newwords, rstrip(multiword))
                end
            end
            multiword = ""
        end
    end
    phrases = collect(Set(newwords))
    final = Vector{Tuple{Float64, SubString{String}}}(undef, length(phrases))
    splittedphrase = split.(phrases)
    for i in eachindex(phrases)
        tempscore::AbstractFloat = 0.0
        if length(splittedphrase[i]) < 2
            final[i] = (toprankscores[findall(x->x==phrases[i], 
            toprankwords)[1]::Integer], phrases[i]::AbstractString)
        else
            for j in eachindex(splittedphrase[i])
                tempscore += toprankscores[findall(x->x==splittedphrase[i][j], 
                toprankwords)[1]::Integer]
            end
            final[i] = (tempscore, phrases[i]::AbstractString)
            tempscore = 0
        end
    end
    map(x -> x[2], sort(final; rev=true))::Vector{SubString{String}}
end

function simulation_baseline_algo()
    dataset = pycall(datasets.load_dataset, PyObject, PyObject("midas/semeval2010"), PyObject("raw"), split="test")
    alldocuments = [dataset[i]["document"] for i=0:convert(Integer, dataset["num_rows"]) - 1]
    goldenkeys = [dataset[i]["extractive_keyphrases"] ∪ dataset[i]["abstractive_keyphrases"] for i=0:convert(Integer, dataset["num_rows"]) - 1]
    #for i in 0:(convert(Integer, dataset["num_rows"]) - 1)
    #for i in 0:3
    #    push!(alldocuments, dataset[i]["document"])
    #    push!(goldenkeys, dataset[i]["extractive_keyphrases"])
    #end
    
    extracted_keywords = map(postprocessing, alldocuments)
    (predicted=extracted_keywords, ground_truth=goldenkeys)
    #CSV.write("data/semeval2010_task5_data.csv", DataFrame(predicted=extracted_keywords, ground_truth=goldenkeys))
end

function simulation_wcoref()
    dataset = pycall(datasets.load_dataset, PyObject, PyObject("midas/semeval2010"), PyObject("raw"), split="test")
    alldocuments = [dataset[i]["document"] for i=0:convert(Integer, dataset["num_rows"]) - 1]
    goldenkeys = [dataset[i]["extractive_keyphrases"] ∪ dataset[i]["abstractive_keyphrases"] for i=0:convert(Integer, dataset["num_rows"]) - 1]
    coref_doc = coreference(map(x -> join(x, ' '), alldocuments))
    extracted_keywords = map(postprocessing, coref_doc)
    (predicted=extracted_keywords, ground_truth=goldenkeys)
end

function process_corefsemeval()
    data = CSV.read("data/semeval2010coref.csv", DataFrame)
    transform!(data, "ground_truth" => ByRow(pyeval) => "ground_truth")
    transform!(data, "predicted" => ByRow(split) => "predicted")
    transform!(data, "predicted" => ByRow(postprocessing) => "predicted")
    CSV.write("data/semeval2010coref_data.csv", data)    
end

function process_marujo()
    data = CSV.read("data/marujo_processed.csv", DataFrame)
    transform!(data, "ground_truth" => ByRow(split) => "ground_truth")
    transform!(data, "predicted" => ByRow(split) => "predicted")
    transform!(data, "predicted" => ByRow(postprocessing) => "predicted")
    CSV.write("data/marujo_processed_data.csv", data)
end

function process_corefmarujo()
    data = CSV.read("data/marujocoref.csv", DataFrame)
    transform!(data, "ground_truth" => ByRow(split) => "ground_truth")
    transform!(data, "predicted" => ByRow(split) => "predicted")
    transform!(data, "predicted" => ByRow(postprocessing) => "predicted")
    CSV.write("data/marujocoref_data.csv", data)
end

function relevant_at_k(pred, gr, k)
    length(pred[k] ∩ gr)
end

function precision_at_k(pred, gr, k)
    length(pred[1:k] ∩ gr)/k
end

function recall_at_k(pred, gr, k)
    length(pred[1:k] ∩ gr)/length(gr)
end

function sum_precision(pred, gr, k=0)
    if k==0
        sum([length(pred[i] ∩ gr[i]) for i=1:length(pred)])/sum(map(length, pred))
    else
        valid_k = filter(x -> length(x) >= k, pred)
        mean(precision_at_k(valid_k[i], gr[i], k) for i=1:length(valid_k))
    end
end

function sum_recall(pred, gr, k=0)
    if k==0
        sum([length(pred[i] ∩ gr[i]) for i=1:length(pred)])/sum(map(length, gr))
    else
        valid_k = filter(x -> length(x) >= k, pred)
        mean(recall_at_k(valid_k[i], gr[i], k) for i=1:length(valid_k))
    end
end

function f1_measure(pred, gr, k=0)
    precision = sum_precision(pred, gr, k)
    recall = sum_recall(pred, gr, k)
    2.00 * (precision * recall)/(precision + recall)
end

function average_precision(y_pred, y_gr, k=0)
end

function baseline_metrics_semeval2010()
    data = CSV.read("data/semeval2010_task5_data.csv", DataFrame)
    transform!(data, "ground_truth" => ByRow(pyeval) => "ground_truth")
    #transform!(a, "ground_truth" => ByRow(Meta.parse) => "ground_truth")
    #transform!(a, "ground_truth" => ByRow(eval) => "ground_truth")
    transform!(data, "predicted" => ByRow(Meta.parse) => "predicted")
    transform!(data, "predicted" => ByRow(eval) => "predicted")

    precision = sum_precision(data.predicted, data.ground_truth)
    recall = sum_recall(data.predicted, data.ground_truth)
    f1_m = f1_measure(data.predicted, data.ground_truth)
    println("Precision: $precision")
    println("Recall: $recall")
    println("F1 Measure: $f1_m")
end

function coref_metrics_semeval2010()
    data = CSV.read("data/semeval2010coref_data.csv", DataFrame)
    
    #transform!(data, "ground_truth" => ByRow(pyeval) => "ground_truth")
    #transform!(data, "predicted" => ByRow(Meta.parse) => "predicted")
    #transform!(data, "predicted" => ByRow(eval) => "predicted")

    precision = sum_precision(data.predicted, data.ground_truth)
    recall = sum_recall(data.predicted, data.ground_truth)
    f1_m = f1_measure(data.predicted, data.ground_truth)
    println("Precision: $precision")
    println("Recall: $recall")
    println("F1 Measure: $f1_m")
end

function baseline_metrics_marujo()
    data = CSV.read("data/marujo_processed_data.csv", DataFrame)

    precision = sum_precision(data.predicted, data.ground_truth)
    recall = sum_recall(data.predicted, data.ground_truth)
    f1_m = f1_measure(data.predicted, data.ground_truth)
    println("Precision: $precision")
    println("Recall: $recall")
    println("F1 Measure: $f1_m")
end

function coref_metrics_marujo()
    data = CSV.read("data/marujocoref_data.csv", DataFrame)

    precision = sum_precision(data.predicted, data.ground_truth)
    recall = sum_recall(data.predicted, data.ground_truth)
    f1_m = f1_measure(data.predicted, data.ground_truth)
    println("Precision: $precision")
    println("Recall: $recall")
    println("F1 Measure: $f1_m")
end