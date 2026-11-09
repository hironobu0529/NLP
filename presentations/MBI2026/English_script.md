# English Script — MBI 2026 S24 (Nakamura)

26 slides · 2109 words · about 18–20 min at 110–120 words per minute

## 0. Opening (69 words, ≈0.6 min)

**Slide 1. Language as a Clinical Interface: NLP-Based Assessment, Illness Monitoring, and Responsible AI in Schizophrenia**

Thank you, Professor Kishimoto. Good afternoon, everyone. I am Hironobu Nakamura, a psychiatrist at the Institute of Science Tokyo. Today I would like to talk about language as a clinical interface: how natural language processing can help us assess and monitor schizophrenia, and what it takes to use this technology responsibly.

**Slide 2. COI Disclosure**

I have no conflicts of interest to disclose, and the study was approved by the relevant ethics committees.

## 1. Introduction (442 words, ≈3.8 min)

**Slide 3. Why I Measure Language**

Let me start with a simple question: why does a psychiatrist measure language? In daily practice, on the ward, in the outpatient clinic, and in my work in epilepsy, the words of our patients are our richest source of information. Yet for more than a century we could only describe them: tangential, impoverished, loosened. Natural language processing now allows us to measure them. Since 2020 I have worked on this question within the UNDERPIN project led by Professor Kishimoto. Today's talk has five parts: a brief history, the basic methods, our study, a map of the field, and finally the promise and the risks of bringing these tools into the clinic.

**Slide 4. Two Histories Converge**

These are two histories that developed in parallel. In psychiatry, Bleuler described the loosening of associations in 1911, and Andreasen later turned thought disorder into a clinician-rated scale. In computer science, natural language processing moved from hand-written rules, through statistical methods and latent semantic analysis, to word embeddings such as Word2Vec in 2013, the Transformer in 2017, and BERT. Since 2020, large language models such as ChatGPT have made language technology part of everyday life. The two lines began to converge around 2007, when latent semantic analysis was used to measure coherence in schizophrenia, and later when speech features predicted the onset of psychosis in high-risk youth. Our work belongs to this most recent phase.

**Slide 5. Conceptual Diagram : Past and Present**

Our study builds a bridge between these two traditions. On the left are classical descriptions from psychopathology and linguistics: poverty of thought, neologisms, incoherence, and loosening of associations. On the right are the NLP methods that can quantify each of them. We treated the classical observations as hypotheses and tested them quantitatively, combining hypothesis-driven and data-driven approaches. For example, poverty of thought can be quantified as the ratio of fillers and empty words; neologisms as words absent from a dictionary; incoherence as syntactic abnormality; and loosening of associations as abnormal semantic similarity between words. In this sense, NLP does not replace psychopathology. It gives its classical concepts a measurable form.

**Slide 6. Introduction**

Previous NLP studies have reported reduced syntactic complexity and reduced semantic coherence in schizophrenia, along with poverty of content, altered cohesion, and changes in pronoun use. Computational models include speech graphs, linked to positive thought disorder, and semantic density, linked to negative thought disorder. However, most studies examined a few features, in small samples, with a single task. We therefore conducted a large-scale, multi-feature analysis, hypothesising that reduced syntactic differentiation and increased semantic proximity are core abnormalities. In addition, almost all of this evidence comes from English. Whether the same abnormalities appear in a language with a very different grammar, such as Japanese, was largely unknown.

## 2. NLP fundamentals (211 words, ≈1.8 min)

**Slide 7. Conceptual Diagram 1: Integrative Approach**

To do this, we first need to decide what to measure. Language can be analysed at four levels. The morphological level covers word classes and word counts. The syntactic level covers sentence structure, such as dependency trees. The semantic level covers how words relate in meaning. And the contextual level covers coherence between sentences. Prior studies usually picked one level; our approach covered all four at once. The important point is that each level corresponds to a classical clinical concept, so that a number on a screen can be translated back into the language of psychopathology.

**Slide 8. Four Families of NLP Models**

Next, we need to decide how to measure it. Roughly, there are four families of models. Rule- and count-based methods, such as dictionaries, parsers, and TF-IDF, tell us which words and structures are used. Static embeddings such as Word2Vec place words in a semantic space, so we can measure how close they are. Contextual models such as BERT and SentenceBERT represent meaning in context, which is useful for coherence. Finally, large language models can rate, summarise, and generate text without task-specific training. Moving to the right, representational power increases, but interpretability decreases. Because our goal was clinical understanding, we deliberately combined the first three families, so that every marker can be explained to a clinician.

## 3. Our study (911 words, ≈7.9 min)

**Slide 9. Material & Method: Participants**

Now to our study, recently published in Psychological Medicine. Within the UNDERPIN project, we recruited 104 patients with schizophrenia and 101 healthy controls from seven hospitals and three clinics in Japan between 2018 and 2024. Participants completed two to five visits, each with three semi-structured tasks: free conversation about daily life; picture description using standardised images; and storytelling, retelling Cinderella or the Japanese folktale Kaguya-hime. Using three tasks matters, because each task places different demands on the speaker. Free conversation is open and spontaneous; picture description is anchored to a visual stimulus; and storytelling requires narrative planning and memory. Interviews lasted thirty to sixty minutes, and interviewers kept a neutral style to minimise their influence on speech.

**Slide 10. Method: Speech Analyses**

All speech was transcribed verbatim, and only the participant's speech was analysed. We extracted 76 features: parts of speech and named entities with the GiNZA parser; syntactic measures such as dependency distance and tree depth; word similarity and semantic networks with Word2Vec and TF-IDF; and sentence similarity and perseveration with SentenceBERT. In total, this gave us a broad profile covering the four levels of language that I introduced earlier.

**Slide 11. Methods: ANALYSIS PIPELINE**

This is our analysis pipeline. First, features were adjusted for speech length and standardised. Second, exploratory group comparisons used generalised estimating equations, or GEEs, to account for repeated visits. Third, because many features are correlated, we used exploratory factor analysis and selected one representative feature per factor. Fourth, we fitted GEE models on the training set. Finally, we evaluated performance on a held-out test set, with every step confined to the training data to prevent information leakage.

**Slide 12. Results: Demographics**

The two groups were similar in age and sex. As expected, patients had higher PANSS scores; the mean total score was 63, indicating mostly mild to moderate illness. Years of education were lower in patients, but were available only for a subset; a sensitivity analysis adjusting for education gave consistent results.

**Slide 13. Methods: ANALYSIS PIPELINE**

First, the exploratory comparisons.

**Slide 14. Results: Exploratory Analyses**

In free conversation, 35 of the 76 features differed significantly after FDR correction, across all levels of language. Here are two examples: patients used more person-related named entities, and fewer adverbs. These exploratory results are descriptive only. They tell us that the abnormalities are widespread, but they do not tell us which ones are essential.

**Slide 15. Results: Exploratory Analyses (Semantic Network)**

Semantic network measures also differed. The clustering coefficient was higher in patients, and it correlated with PANSS negative symptoms and with a PANSS item on the fluency of speech.

**Slide 16. Semantic Network Analysis Using a TF-IDF Framework**

Briefly, here is how these networks are built. Using TF-IDF, we pick each speaker's most characteristic content words, connect the words whose Word2Vec similarity exceeds a threshold, and compute graph measures such as the clustering coefficient, diameter, average distance, and closeness centrality. By eye, a patient's network and a control's network look similar, so we rely on the numbers. The idea behind this analysis is simple: if a speaker keeps returning to closely related words, the network becomes tight and clustered; if a speaker moves freely between topics, the network becomes wider.

**Slide 17. Summary of Results: Semantic Network Analysis**

In schizophrenia, clustering and density increased, while average distance, diameter, and closeness centrality decreased. In other words, the semantic network is locally dense but globally poorly integrated: patients tend to stay within close semantic neighbourhoods and rarely travel to distant ones.

**Slide 18. Methods: ANALYSIS PIPELINE**

With so many features differing, we asked which dimensions are truly essential. To answer this in an interpretable way, we reduced the features with factor analysis.

**Slide 19. Result: Factor analysis & GEE (Task 1)**

In free conversation, factor analysis gave a stable nine-factor solution, and we kept one representative feature per factor, with low collinearity. In the multivariable GEE model, three features remained significant. First, higher mean pairwise word similarity: a one-standard-deviation increase was associated with about a 2.3-fold increase in the odds of schizophrenia. Second and third, lower ratios of case particles and adverbs. In Japanese, case particles such as ga, o, and ni mark who did what to whom. In the independent test set, the model reached an AUC of 0.87, with a sensitivity of 0.76 and a specificity of 0.90. Notably, among the major parts of speech, only adverbs were significantly reduced, not nouns, verbs, or adjectives. In Japanese, adverbs modify not only verbs but whole sentences, adding manner, degree, and the speaker's stance.

**Slide 20. Result: Comparison of Tasks**

We then applied the same pipeline to storytelling and picture description. Adverbial features were significant in every task, and case particles were reduced in a task-invariant way. Semantic-network and discourse features also contributed across tasks. Interestingly, perseveration was strongly elevated in storytelling, mirroring the clinical phenomenon of perseveration. Sensitivity analyses showed that the adverb effect was strongest in free conversation, whereas the reduction in case particles did not depend on the task. This suggests that case-particle reduction reflects a stable, task-general simplification, while reduced modification is most visible when speech is spontaneous.

**Slide 21. Discussion**

Taken together, we propose a tripartite linguistic phenotype of schizophrenia. First, reduced morphosyntactic explicitness: fewer case particles, so less overt marking of who did what to whom. Second, semantic narrowing: words drawn from close neighbourhoods. Third, weakened contextual modulation: fewer adverbs, which add manner, degree, and nuance. These dimensions are separable but interacting, and they go beyond the positive-negative dichotomy of thought disorder. Similar findings in Turkish, another agglutinative language, suggest shared vulnerabilities whose expression depends on language type, and the semantic findings parallel neuroimaging studies of network integration. Clinically, these markers are simple, interpretable, and related to PANSS severity, so they could serve as accessible proxies of illness burden. But these interpretations remain hypotheses that need to be tested with cognitive and functional measures.

## 4. Trends and the map (104 words, ≈0.9 min)

**Slide 22. Mapping the Field**

Let me step back and place this work on a map of the field. The horizontal axis runs from interpretable features to large models; the vertical axis runs from cross-sectional diagnosis to longitudinal monitoring and intervention. Our study sits here: interpretable and cross-sectional. The field is moving up and to the right: from single features to multidimensional profiles, from English to many languages, from the laboratory to smartphones, from text to multimodal data including voice and wearables, as Dr. Liang has just shown, and from handcrafted features to large language models. Our own next step is upward: longitudinal monitoring with the same interpretable markers.

## 5. Prospects and challenges (298 words, ≈2.6 min)

**Slide 23. Limitations: What Is Missing**

Before moving to the clinic, we must be honest about what is still missing. Our sample had established, mostly mild to moderate illness, with no clinical high-risk group, so we cannot predict onset. All data were in Japanese, so cross-linguistic validation is needed. We relied on PANSS rather than dedicated thought-disorder scales, and on spoken language only. Finally, our semantic measures were based on a single Word2Vec model trained on Wikipedia, so they should be replicated with other embeddings, including transformer-based models.

**Slide 24. Promise and Peril**

So, what are the promises and the perils? On the promise side: objective assessment of severity, early warning of relapse, tracking of treatment response, continuous monitoring between visits, and less variability between raters. On the peril side: voice and words can identify a person, so privacy is critical, and patients may feel watched. Models can be biased across languages, dialects, and levels of education. Large models are opaque and can hallucinate, and generative AI may even echo or reinforce delusional content. There is also a risk of overdiagnosis, and of clinicians trusting the machine too much. Language data are intimate data: the same signal that informs care can also expose the person. In psychiatry this matters even more, because the content of speech is often about the most private parts of a person's life: their fears, their beliefs, and their relationships.

**Slide 25. Toward Responsible Use**

Responsible implementation rests on three pillars. Evidence: external, cross-linguistic, longitudinal, and prospective validation, including high-risk cohorts and links to real-world function. Ethics and governance: privacy by design, consent that can be withdrawn, audits for bias, transparency, and alignment with medical-device and AI regulations. And clinical integration: humans stay in the loop, clinicians and patients are educated, patients and families help design the tools, and success is measured by benefit to care, not by AUC alone.

## 6. Conclusion (74 words, ≈0.6 min)

**Slide 26. Conclusion: Take-Home Messages**

To conclude, I have three messages. First, language is a clinical interface, and NLP can turn it into measurable, interpretable markers. Second, in Japanese speech, schizophrenia shows a tripartite phenotype, with fewer case particles, semantic narrowing, and fewer adverbs, giving strong discrimination and cross-task consistency. Third, the path to the clinic runs through validation and responsibility: AI should augment, not replace, clinical understanding, the therapeutic relationship, and human responsibility. Thank you for your attention.
