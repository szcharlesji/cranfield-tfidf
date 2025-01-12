import string
from stop_list import closed_class_stop_words
import math
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

nltk.download('wordnet')

def read_query():
    query_file = open("cran.qry", "r")
    queries = query_file.read()
    query_file.close()

    # Parse query
    queries = queries.split(".I")
    parsed_queries = []

    for query in queries[1:]:
        index_start = query.find(" ") + 1
        index_end = query.find("\n")
        index = int(query[index_start:index_end])
        text_start = query.find(".W\n") + 3
        text_end = query.find(".I", text_start)
        if text_end == -1:
            text_end = len(query)
        text = query[text_start:text_end]

        text = text.replace("\n", " ")

        # Remove punctuation:
        translator = str.maketrans("", "", string.punctuation)
        text = text.translate(translator)

        parsed_queries.append([index, text])

    return parsed_queries


def read_abstracts():
    abstracts = open("cran.all.1400", "r").read()
    abstracts = abstracts.split(".I")
    parsed_abstracts = []

    for abstract in abstracts[1:]:
        components = abstract.split("\n.")
        data = {}

        # Extract the Index
        index_start = abstract.find(" ") + 1
        index_end = abstract.find("\n")
        data["I"] = abstract[index_start:index_end]

        for component in components:
            if not component:
                continue

            label = component[0]
            content = component[2:].lstrip()
            if label == "T" or label == "W":
                # Replace newlines and remove punctuation (optional)
                content = content.replace("\n", " ")
                translator = str.maketrans("", "", string.punctuation)
                content = content.translate(translator)

            data[label] = content

        parsed_abstracts.append(data)

    return parsed_abstracts

# def morpheme(word):
#     if word.endswith("s"):
#         word = word[:-1]
#     if word.endswith("ly"):
#         word = word[:-2]
#     if word.endswith("al"):
#         word = word[:-2]
#     if word.endswith("ion"):
#         word = word[:-3]
#     if word.endswith("ed"):
#         word = word[:-2]
#     elif word.endswith("ing"):
#         word = word[:-3]
        
#     return word



def morpheme(word):
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word)
    if lemma == word:  # If no change from lemmatization
        stemmer = PorterStemmer()
        return stemmer.stem(word)
    else:
        return lemma


word_counts = {}


def calculate_idf(queries, abst=False):
    """Calculates the IDF scores for each word in the queries."""
    if abst:
        documents = [query["W"] for query in queries]
    else:
        documents = [query[1] for query in queries]

    for doc in documents:
        words = doc.translate(str.maketrans("", "", string.punctuation)).lower().split()
        words = [
            word
            for word in words
            if word not in closed_class_stop_words and not word.isdigit()
        ]
        seen = set()

        if not abst:
            for word in words:
                # merge singular and plural forms
                word = morpheme(word)
                word_counts[word] = word_counts.get(word, 0) + 1
                if word not in seen:
                    seen.add(word)

    idf = {}
    N = len(documents)  # Total number of documents
    for word, count in word_counts.items():
        idf[word] = math.log(N / count)  # Standard IDF calculation

    return idf


def calculate_tf_idf(queries, idf):
    """Calculates TF-IDF scores for each query."""
    for query in queries:
        tf = {}  # Term frequencies for this query
        words = (
            query[1]
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
            .split()
        )
        words = [
            word
            for word in words
            if word not in closed_class_stop_words and not word.isdigit()
        ]

        for word in words:
            word = morpheme(word)
            tf[word] = tf.get(word, 0) + 1

        # query["vector"] = [tf.get(word, 0) * idf.get(word, 0) for word in idf]
        query.append([])
        for word in idf:
            vector = tf.get(word, 0) * idf.get(word, 0)
            vector = 0 if vector == 0 else 1 + math.log(vector)
            query[2].append([word, vector])
            # print(word, vector)

    return queries


def calculate_abstract_vectors(abstracts, idf_abstracts):
    """Calculates TF-IDF vectors for each abstract."""

    for abstract in abstracts:
        tf = {}
        words = (
            abstract["W"]
            .translate(str.maketrans("", "", string.punctuation))
            .lower()
            .split()
        )
        words = [
            word
            for word in words
            if word not in closed_class_stop_words and not word.isdigit()
        ]

        for word in words:
            word = morpheme(word)
            tf[word] = tf.get(word, 0) + 1

        abstract["vector"] = []
        for word in idf_abstracts:
            vector = tf.get(word, 0) * idf_abstracts.get(word, 0)
            vector = 0 if vector == 0 else 1 + math.log(vector)
            abstract["vector"].append(vector)

    return abstracts


def calculate_cosine_similarity(query_vector, abstract_vector):
    """Calculates the cosine similarity between two vectors."""
    similarity = 0
    dot_product = 0
    magnitude_query = 0
    magnitude_abstract = 0

    for i, query_value in enumerate(query_vector):
        dot_product += query_value * abstract_vector[i]
        magnitude_query += query_value**2
        magnitude_abstract += abstract_vector[i] ** 2

    magnitude_query = math.sqrt(magnitude_query)
    magnitude_abstract = math.sqrt(magnitude_abstract)
    if magnitude_abstract:
        similarity = dot_product / (magnitude_query * magnitude_abstract)
    else:
        similarity = 0

    return similarity


def main():
    queries = read_query()
    idf = calculate_idf(queries)
    tf_idf_query = calculate_tf_idf(queries, idf)

    abstracts = read_abstracts()
    idf_abstracts = calculate_idf(abstracts, True)
    abstracts = calculate_abstract_vectors(abstracts, idf_abstracts)

    for index, query in enumerate(tf_idf_query):
        similarities = []
        for abstract in abstracts:
            vec_query = [val[1] for val in query[2]]
            abs_query = abstract["vector"]
            similarity = calculate_cosine_similarity(vec_query, abs_query)
            similarities.append((abstract["I"], similarity))

        # Sort by descending cosine similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Output in required format
        for abstract_index, similarity_score in similarities:
            if similarity_score > 0:
                print(index + 1, abstract_index, similarity_score)
                
        if sum([sim[1] for sim in similarities]) == 0:
            count = 0
            for abstract_index, similarity_score in similarities:
                print(index + 1, abstract_index, similarity_score)
                count += 1
                if count == 100:
                    break


if __name__ == "__main__":
    main()
