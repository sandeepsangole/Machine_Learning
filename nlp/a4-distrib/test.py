import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Your paragraph
paragraph = [{'title': 'Marianne McAndrew', 'text': '<s>Marianne McAndrew Marianne Christine McAndrew is an actress known for her role as Irene Molloy in the film "Hello, Dolly!"</s><s>Career. "Hello, Dolly!" was McAndrew\'s first credited film role. The role of Irene Molloy was given considerably more attention in the film than in earlier Broadway productions. "Hello, Dolly!" earned McAndrew two Golden Globe nominations in 1969; Golden Globe Award for Best Supporting Actress – Motion Picture and the since discontinued Golden Globe Award for New Star of the Year – Actress, as well as generally good reviews. She landed a starring (second billed) role in her next film ("The Seven Minutes"). By 1971, she had made appearances in popular TV shows including "Hawaii 5-0", "Mannix", "Cannon" and "Love, American Style" One notable later film role was her co-starring role in "The Bat People", with her husband Stewart Moss. The film was widely panned, but is still somewhat known today as a "bad film". "The Bat People" was also her last film released in theaters; she has only worked in television since its release. Her only other later role of particular note is her role of'}, {'title': 'Marianne McAndrew', 'text': ' Doris Williams in "Growing Up Brady", a TV film about the popular show. McAndrew, along with everything else related to the film "Hello, Dolly!" experienced something of a resurgence in popularity with the release of "WALL-E", which featured clips from the film, including a duet with McAndrew\'s character (but McAndrew did not do her own singing in the film, which some news outlets claimed in articles about "WALL-E").</s><s>Personal life. McAndrew married actor Stewart Moss in 1968, remaining married until his death in 2017. They starred together in "The Bat People". McAndrew has two brothers.</s><s>Awards and honors. "Hello, Dolly!" earned McAndrew two Golden Globe nominations in 1969; Golden Globe Award for Best Supporting Actress – Motion Picture and the since discontinued Golden Globe Award for New Star of the Year – Actress.</s><s>References.</s>'}]

# Extract the text from the paragraph
text = paragraph[0]['text']

# Process the text with spaCy to split it into sentences
doc = nlp(text)

# Extract the sentences
sentences = [sent.text for sent in doc.sents]

# Print the sentences
for sentence in sentences:
    print(sentence)
    print("======")
