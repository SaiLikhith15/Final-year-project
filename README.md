# Implementation of Novel Machine Learning Methods for Analysis and Detection of Fake Reviews in Social Media

# ABSTRACT

With the continual evolution, online evaluations are increasingly seen as a critical aspect in establishing and keeping a positive reputation. Furthermore, they play an important part in the decision-making process for end consumers. A good review for a specific object typically draws more consumers and leads to a significant rise in sales. Deceptive or phoney evaluations are now purposefully generated in order to develop a virtual reputation and attract potential clients. As a result, detecting false reviews is an active and ongoing research topic. Identifying phoney reviews is dependent not only on the essential elements of the reviews, but also on the reviewers' behaviour. This research provides a machine learning method for detecting false reviews. In addition to the review features extraction approach, this research employs different features engineering techniques to extract diverse reviewer behaviours. The study examines the performance of machine learning classifiers; Decision Tree, Random Forest, SVC, CNN, Naïve Bias, Logistic Regression, XGBoost, ANN, LSTM, BERT. The results demonstrate that the algorithm is better at determining whether a review is bad or genuine.  


Keywords: Decision Tree, Random Forest, SVC, CNN, Naïve Bias, Logistic Regression, XGBoost, ANN, LSTM, BERT.

# INTRODUCTION

Nowadays, when customers want to draw a decision about services or products, reviews become the main source of their information. For example, when customers take the initiation to book a hotel, they read the reviews on the opinions of other customers on the hotel services. Depending on the feedback of the reviews, they decide to book room or not. If they came to a positive feedback from the reviews, they probably proceed to book the room. Thus, historical reviews became very credible sources of information to most people in several online services. Since, reviews are considered forms of sharing authentic feedback about positive or negative services, any attempt to manipulate those reviews by writing misleading or inauthentic content is considered as deceptive action and such reviews are labelled as fake Such case leads us to think what if not all the written reviews are honest or credible. What if some of these reviews are fake. Thus, detecting fake review has become and still in the state of active and required research area.

The rise of social media has blurred the line between authentic content and advertising, leading to an explosion in deceptive endorsements across the marketplace. Fake online reviews and other deceptive endorsements often tout products throughout the online world. Consequently, the FTC is now using its Penalty Offense Authority to remind advertisers of the law and deter them from breaking it. By sending a Notice of Penalty Offenses to more than 700 companies, the agency is placing them on notice they could incur significant civil penalties—up to $43,792 per violation—if they use endorsements in ways that run counter to prior FTC administrative cases.
“Fake reviews and other forms of deceptive endorsements cheat consumers and undercut honest businesses,” said Samuel Levine, Director of the FTC’s Bureau of Consumer Protection. “Advertisers will pay a price if they engage in these deceptive practices.”

The Notice of Penalty Offenses allows the agency to seek civil penalties against a company that engages in conduct that it knows has been found unlawful in a previous FTC administrative order, other than a consent order.
The Notice sent to the companies outlines a number of practices that the FTC determined to be unfair or deceptive in prior administrative cases. These include, but are not limited to: falsely claiming an endorsement by a third party; misrepresenting whether an endorser is an actual, current, or  recent user; using an endorsement to make deceptive performance claims; failing to disclose an unexpected material connection with an endorser; and misrepresenting that the experience of endorsers represents consumers’ typical or ordinary experience.

Companies receiving the notice represent an array of large companies, top advertisers, leading retailers, top consumer product companies, and major advertising agencies. A full list of the businesses receiving the Notice from the FTC is available on the FTC’s website. A recipient’s presence on this list does not in any way suggest that it has engaged in deceptive or unfair conduct. In addition to the Notice, the FTC has created multiple resources for business to ensure that they are following the law when using endorsements to advertise their products and services, which can be found on the FTC’s website.

To this end, this paper applies several machine learning classifiers to identify fake reviews based on the content of the reviews as well as several extracted features from the reviewers. We apply the classifiers on real corpus of reviews taken from open source sites. Besides the normal natural language processing on the corpus to extract and feed the features of the reviews to the classifiers, the paper also applies several features engineering on the corpus to extract various behaviours of the reviewers. The paper compares the impact of extracted features of the reviewers if they are taken into consideration within the classifiers. The papers compares the results in the absence and the presence of the extracted features in two different language models namely TF-IDF. The results indicates that the engineered features increase the performance of fake reviews detection process.The rapid growth of the Internet influenced many of our daily activities. One of the very rapid growth area is 
ecommerce. Generally e-commerce provide facility for customers to write reviews related with its service. The existence of these reviews can be used as a source of information. For examples, companies can use it to make design decisions of their products or services, while potential customers can use it to decide either to buy or to use a product. Unfortunately, the importance of the review is misused by certain parties who tried to create fake reviews, both aimed at raising the popularity or to discredit the product. This research aims to detect fake reviews for a product by using the text and rating property from a review.

The rapid growth of the Internet influenced many of our daily activities. One of the very rapid growth area is ecommerce. Generally e-commerce provide facility for customers to write reviews related with its service. The existence of these reviews can be used as a source of information. For examples,companies can use it to make design decisions of their products or services, while potential customers can use it to decide either to buy or to use a product. Unfortunately, the importance of the review is misused by certain parties who tried to create fake reviews, both aimed at raising the popularity or to discredit the product. This research aims to detect fake reviews for a product by using the text and rating property from a review.The rapid growth of the Internet influenced many of our daily activities. One of the very rapid growth area is ecommerce. Generally e-commerce provide facility for customers to write reviews related with its service. The existence of these reviews can be used as a source of information. For examples, companies can use it to make design decisions of their products or services, while potential customers can use it to decide either to buy or to use a product. 

Unfortunately, the importance of the review is misused by certain parties who tried to create fake reviews, both aimed at raising the popularity or to discredit the product. This research aims to detect fake reviews for a product by using the text and rating property from a review. Machine learning techniques can provide a big contribution to detect fake reviews of web contents. Generally, web mining techniques find and extract useful information using several machine learning algorithms. One of the web mining tasks is content mining. A traditional example of content mining is opinion mining which is concerned of finding the sentiment of text (positive or negative) by machine learning where a classifier is trained to analyse the features of the reviews together with the sentiments. Usually, fake reviews detection depends not only on the category of reviews but also on certain features that are not directly connected to the content. Building features of reviews normally involves text and natural language processing NLP. However, fake reviews may require building other features linked to the reviewer himself like for example review time/date or his writing styles.

Thus the successful fake reviews detection lies on the construction of meaningful features extraction of the reviewers. Usually, fake reviews detection depends not only on the category of reviews but also on certain features that are not directly connected to the content. Building features of reviews normally involves text and natural language processing NLP. However, fake reviews may require building other features linked to the reviewer himself like for example review time/date or his writing styles. Thus the successful fake reviews detection lies on the construction of meaningful features extraction of the reviewers.


# STATEMENT OF THE PROBLEM

The problem addressed in this article is the proliferation of fake reviews in social media and the need to identify and differentiate them from genuine reviews. Fake reviews can be used to deceive consumers, misrepresent products and services, and harm businesses. Manual identification of fake reviews is a time-consuming and labor-intensive process, which is not scalable for large datasets. Therefore, there is a need for novel machine learning methods that can automatically analyze and detect fake reviews in social media with high accuracy.


# OBJECTIVE

1. Developing a robust and scalable machine learning model that can analyze and detect fake reviews in social media with high accuracy. Exploring various techniques, such as sentiment analysis, natural language processing, and deep learning, to identify the underlying patterns and characteristics of fake reviews.

2. Evaluating the effectiveness of the proposed solution in terms of accuracy, scalability, and practicality. Identifying and addressing the challenges related to data bias, privacy, and the dynamic nature of social media.

3. Providing businesses and consumers with reliable information to make informed decisions about products and services based on genuine reviews.\

# SCOPE

Implementing the system in a scalable and efficient manner that can handle large volumes of data. Evaluating the effectiveness of the system in terms of accuracy, precision, recall, and F1 score. Addressing ethical and legal concerns related to the collection and use of user-generated data for training and testing the models. Integrating the system with social media platforms to provide real-time analysis and detection of fake reviews.
 

![image](https://github.com/SaiLikhith15/Final-year-project/assets/121685647/5cc01c2a-5b5b-4fd1-8756-474ca50c9081)

# Input Design:
In an information system, input is the raw data that is processed to produce output. During the input design, the developers must consider the input devices such as PC, MICR, OMR, etc.
Therefore, the quality of system input determines the quality of system output. Well-designed input forms and screens have following properties −

It should serve specific purpose effectively such as storing, recording, and retrieving the information.
It ensures proper completion with accuracy.
It should be easy to fill and straightforward.
It should focus on user’s attention, consistency, and simplicity.
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
The design of output is the most important task of any system. During output design, developers identify the type of outputs needed, and consider the necessary output controls and prototype report layouts.

Objectives of Output Design:
The objectives of input design are:
To develop output design that serves the intended purpose and eliminates the production of unwanted output.
To develop the output design that meets the end user’s requirements.
To deliver the appropriate quantity of output.
To form the output in appropriate format and direct it to the right person.
To make the output available on time for making good decisions.






