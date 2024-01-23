# Implementation of Novel Machine Learning Methods for Analysis and Detection of Fake Reviews in Social Media

# ABSTRACT

With the continual evolution, online evaluations are increasingly seen as a critical aspect in establishing and keeping a positive reputation. Furthermore, they play an important part in the decision-making process for end consumers. A good review for a specific object typically draws more consumers and leads to a significant rise in sales. Deceptive or phony evaluations are now purposefully generated to develop a virtual reputation and attract potential clients. As a result, detecting false reviews is an active and ongoing research topic. Identifying phony reviews is dependent not only on the essential elements of the reviews but also on the reviewers' behavior. This research provides a machine-learning method for detecting false reviews. In addition to the review features extraction approach, this research employs different feature engineering techniques to extract diverse reviewer behaviors. The study examines the performance of machine learning classifiers; Decision Tree, Random Forest, SVC, CNN, Naïve Bias, Logistic Regression, XGBoost, ANN, LSTM, and BERT. The results demonstrate that the algorithm is better at determining whether a review is bad or genuine.  


Keywords: Decision Tree, Random Forest, SVC, CNN, Naïve Bias, Logistic Regression, XGBoost, ANN, LSTM, BERT.

# INTRODUCTION

Nowadays, when customers want to decide on services or products, reviews become the main source of their information. For example, when customers take the initiation to book a hotel, they read the reviews on the opinions of other customers on the hotel services. Depending on the feedback from the reviews, they decide to book a room or not. If they came to positive feedback from the reviews, they probably proceeded to book the room. Thus, historical reviews became very credible sources of information to most people in several online services. Since reviews are considered forms of sharing authentic feedback about positive or negative services, any attempt to manipulate those reviews by writing misleading or inauthentic content is considered as deceptive action, and such reviews are labeled as fake Such case leads us to think about what if not all the written reviews are honest or credible. What if some of these reviews are fake? Thus, detecting fake reviews has become and is still in the state of an active and required research area.

The rise of social media has blurred the line between authentic content and advertising, leading to an explosion in deceptive endorsements across the marketplace. Fake online reviews and other deceptive endorsements often tout products throughout the online world. Consequently, the FTC is now using its Penalty Offense Authority to remind advertisers of the law and deter them from breaking it. By sending a Notice of Penalty Offenses to more than 700 companies, the agency is placing them on notice they could incur significant civil penalties—up to $43,792 per violation—if they use endorsements in ways that run counter to prior FTC administrative cases.
“Fake reviews and other forms of deceptive endorsements cheat consumers and undercut honest businesses,” said Samuel Levine, Director of the FTC’s Bureau of Consumer Protection. “Advertisers will pay a price if they engage in these deceptive practices.”

The Notice of Penalty Offenses allows the agency to seek civil penalties against a company that engages in conduct that it knows has been found unlawful in a previous FTC administrative order, other than a consent order.
The Notice sent to the companies outlines several practices that the FTC determined to be unfair or deceptive in prior administrative cases. These include but are not limited to falsely claiming an endorsement by a third party; misrepresenting whether an endorser is an actual, current, or  recent user; using an endorsement to make deceptive performance claims; failing to disclose an unexpected material connection with an endorser; and misrepresenting that the experience of endorsers represents consumers’ typical or ordinary experience.

Companies receiving the notice represent an array of large companies, top advertisers, leading retailers, top consumer product companies, and major advertising agencies. A full list of the businesses receiving the Notice from the FTC is available on the FTC’s website. A recipient’s presence on this list does not in any way suggest that it has engaged in deceptive or unfair conduct. In addition to the Notice, the FTC has created multiple resources for businesses to ensure that they are following the law when using endorsements to advertise their products and services, which can be found on the FTC’s website.

To this end, this paper applies several machine learning classifiers to identify fake reviews based on the content of the reviews as well as several extracted features from the reviewers. We apply the classifiers to the real corpus of reviews taken from open-source sites. Besides the normal natural language processing on the corpus to extract and feed the features of the reviews to the classifiers, the paper also applies several features engineering on the corpus to extract various behaviors of the reviewers. The paper compares the impact of extracted features of the reviewers if they are taken into consideration within the classifiers. The papers compare the results in the absence and the presence of the extracted features in two different language models namely TF-IDF. The results indicate that the engineered features increase the performance of the fake reviews detection process. The rapid growth of the Internet influenced many of our daily activities. One of the very rapid growth areas is 
e-commerce. Generally, e-commerce provides a facility for customers to write reviews related to its service. The existence of these reviews can be used as a source of information. For example, companies can use it to make design decisions for their products or services, while potential customers can use it to decide either to buy or to use a product. Unfortunately, the importance of the review is misused by certain parties who try to create fake reviews, both aimed at raising popularity or discredit the product. This research aims to detect fake reviews for a product by using the text and rating property from a review.

The rapid growth of the Internet influenced many of our daily activities. One of the very rapid growth areas is e-commerce. Generally, e-commerce provides a facility for customers to write reviews related to its service. The existence of these reviews can be used as a source of information. For example, companies can use it to make design decisions for their products or services, while potential customers can use it to decide either to buy or to use a product. Unfortunately, the importance of the review is misused by certain parties who try to create fake reviews, both aimed at raising popularity or discredit the product. This research aims to detect fake reviews for a product by using the text and rating property from a review. The rapid growth of the Internet influenced many of our daily activities. One of the very rapid growth areas is e-commerce. Generally, e-commerce provides a facility for customers to write reviews related to its service. The existence of these reviews can be used as a source of information. For example, companies can use it to make design decisions for their products or services, while potential customers can use it to decide either to buy or to use a product. 

Unfortunately, the importance of the review is misused by certain parties who try to create fake reviews, both aimed at raising popularity or discredit the product. This research aims to detect fake reviews for a product by using the text and rating property from a review. Machine learning techniques can provide a big contribution to detecting fake reviews of web content. Generally, web mining techniques find and extract useful information using several machine learning algorithms. One of the web mining tasks is content mining. A traditional example of content mining is opinion mining which is concerned with finding the sentiment of text (positive or negative) by machine learning where a classifier is trained to analyse the features of the reviews together with the sentiments. Usually, fake reviews detection depends not only on the category of reviews but also on certain features that are not directly connected to the content. Building features of reviews normally involves text and natural language processing NLP. However, fake reviews may require building other features linked to the reviewer himself for example review time/date or his writing style.

Thus the successful fake reviews detection lies in the construction of meaningful feature extraction of the reviewers. Usually, fake reviews detection depends not only on the category of reviews but also on certain features that are not directly connected to the content. Building features of reviews normally involves text and natural language processing NLP. However, fake reviews may require building other features linked to the reviewer himself for example review time/date or his writing style. Thus the successful fake review detection lies in the construction of meaningful feature extraction of the reviewers.


# STATEMENT OF THE PROBLEM

The problem addressed in this article is the proliferation of fake reviews in social media and the need to identify and differentiate them from genuine reviews. Fake reviews can deceive consumers, misrepresent products and services, and harm businesses. Manual identification of fake reviews is a time-consuming and labor-intensive process, which is not scalable for large datasets. Therefore, there is a need for novel machine-learning methods that can automatically analyze and detect fake reviews on social media with high accuracy.


# OBJECTIVE

1. Developing a robust and scalable machine learning model that can analyze and detect fake reviews in social media with high accuracy. Exploring various techniques, such as sentiment analysis, natural language processing, and deep learning, to identify the underlying patterns and characteristics of fake reviews.

2. Evaluating the effectiveness of the proposed solution in terms of accuracy, scalability, and practicality, identifying and addressing the challenges related to data bias, privacy, and the dynamic nature of social media.

3. Providing businesses and consumers with reliable information to make informed decisions about products and services based on genuine reviews.\

# SCOPE

Implementing the system in a scalable and efficient manner that can handle large volumes of data. Evaluating the effectiveness of the system in terms of accuracy, precision, recall, and F1 score. Addressing ethical and legal concerns related to the collection and use of user-generated data for training and testing the models. Integrating the system with social media platforms to provide real-time analysis and detection of fake reviews.
 

![image](https://github.com/SaiLikhith15/Final-year-project/assets/121685647/5cc01c2a-5b5b-4fd1-8756-474ca50c9081)

# Comparison of Proposed System With Existing Algorithms:

The evaluated algorithms for the detection of fake reviews are Random forest, KNN, SVM, BERT, CNN, XG Boost, ANN, LSTM, Decision tree,  and Logistic Regression. Classification Accuracy, Precision, Recall, and F-Measure are various parameters used to verify the algorithms. The proposed methods are evaluated using MATLAB R2018 edition. After careful evaluation of all the algorithms in MATLAB as per the parameters considered the evaluated outputs are shown in Tables 3 &4

Evaluated Results of Various Algorithms for the detection of fake reviews
S. No.	Model	Precision Score
(%)	Recall 
Score
(%)	F1 
Score (%)
1	ANN	68.59	74.56	65.23
2	CNN	72.59	75.25	68.89
3	Logistic Regression	74.26	78.78	71.25
4	Support Vector Machine	75.28	79.12	74.89
5	Gaussian Naive Bayes	78.28	82.15	76.25
6	K-Nearest Neighbor	80.27	86.45	78.59
7	Decision tree	83.54	88.78	82.45
8	Random Forest	86.54	89.45	83.78
9	Stochastic Gradient Descent (SGD)	88.78	90.25	85.45
10	Proposed Multinomial NB(Proposed)	90.25	93.25	86.52

Evaluated Results of Various Algorithms for Fake News Detection.
![image](https://github.com/SaiLikhith15/Final-year-project/assets/121685647/6a6a9bc4-f823-4940-8f23-f677e75b7fde)

# Comparison of Various Models in terms of Training and Testing Accuracy
Model	Training accuracy
(%)	Testing
Accuracy (%)
Bidirectional LSTM + GLoVe(50D)	91.24	82.56
LSTM + GLoVe(100D)	92.56	84.56
CNN + LSTM + Doc2Vec +TF-IDF	93.45	85.48
CNN+ Attention+ GLoVe(100D)	94.58	86.75
Bi-LSTM + Attention + GLoVe(100D)	95.36	87.45
CNN + Bi-LSTM + Attention + GLoVe(100D)	96.58	88.56
Logistic Regression + TF-IDF(Proposed)	98.54	89.52


# Input Design:
In an information system, input is the raw data that is processed to produce output. During the input design, the developers must consider the input devices such as PC, MICR, OMR, etc.
Therefore, the quality of the system input determines the quality of the system output. Well-designed input forms and screens have the following properties −

It should serve specific purposes effectively such as storing, recording, and retrieving the information.
It ensures proper completion with accuracy.
It should be easy to fill and straightforward.
It should focus on the user’s attention, consistency, and simplicity.
All these objectives are obtained using the knowledge of basic design principles regarding −

What are the inputs needed for the system?
How end users respond to different elements of forms and screens.

Objectives for Input Design:
The objectives of input design are −
To design data entry and input procedures
To reduce input volume
To design source documents for data capture or devise other data capture methods
To design input data records, data entry screens, user interface screens, etc.
To use validation checks and develop effective input controls.

# Output Design:
The design of output is the most important task of any system. During output design, developers identify the type of outputs needed and consider the necessary output controls and prototype report layouts.

Objectives of Output Design:
The objectives of input design are:
To develop an output design that serves the intended purpose and eliminates the production of unwanted output.
To develop the output design that meets the end user’s requirements.
To deliver the appropriate quantity of output.
To form the output in the appropriate format and direct it to the right person.
To make the output available on time for making good decisions.






