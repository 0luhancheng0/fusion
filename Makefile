submission:
	tar -cf - --exclude="embeddings.pt" \
		logs/results.json \
		logs/all_results.json \
		logs/arxiv_hard_negatives.pt \
		logs/arxiv_uniform_negatives.pt \
		logs/ASGC \
		logs/Node2VecLightning \
		logs/TextualEmbeddings \
		$$(find logs/*Fusion -path "*/32" -type d) | pigz -p 4 > submissions.tar.gz

setup:
	tar -xf submissions.tar.gz
