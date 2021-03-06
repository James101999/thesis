using JSON3
using EzXML
using DataFrames
using CSV

function process_keys()
	json_string = read("data/marujo-2012-pre/references/test.reader.json", String)
	JSON3.read(json_string)
end

function process_keys_stemmed()
	json_string = read("data/marujo-2012-pre/references/test.reader.stem.json", String)
	JSON3.read(json_string)
end

function process_data()
	xml_files = [x for x in readdir(joinpath("data/marujo-2012-pre/test"))]
	xml_files_path = map(x -> "data/marujo-2012-pre/test/" * x, xml_files)
	key = hcat(split.(xml_files, ".xml")...)[1, :]
	true_values = process_keys()
	true_values_stemmed = process_keys_stemmed()
	docs = map(readxml, xml_files_path)
	rts = map(root, docs)
	f_all(x) = findall("document/sentences/sentence/tokens/token/word", x)
	output = Vector{Tuple{Vector{String}, Vector{String}, Vector{String}}}(undef, length(rts))
	for i in eachindex(rts)
           output[i] = (map(nodecontent, f_all(rts[i])), [(true_values[key[i]]...)...], 
		   [(true_values_stemmed[key[i]]...)...])
    end
	output
	#all_documents = map(firstelement, rts)
	#sentences = map(firstelement, all_documents)
	#individual_sentence = map(elements, sentences)
end

export_datum = process_data()
#df = DataFrame(Dict(:"predicted"=>export_datum[1], "ground_truth"=>export_datum[2]))
df = DataFrame(predicted = [x[1] for x in export_datum],
 			ground_truth = [x[2] for x in export_datum],
	ground_truth_stemmed = [x[3] for x in export_datum])
transform!(df, "predicted"=>ByRow(x->join(x, ' '))=>"predicted")
CSV.write("data/marujo_processed.csv", df)
#to_csv(df, "data/marujo_processed.csv")
