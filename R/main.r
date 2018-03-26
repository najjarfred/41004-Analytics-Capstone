nswhouse <- read.table("./house_valuation.csv",
   sep=",", header=TRUE, fill=TRUE, quote="")

#Randomly shuffle the data
nswshuffled<-nswhouse[sample(nrow(nswhouse)),]
#Create 10 equally size folds
folds <- cut(seq(1,nrow(nswshuffled)),breaks=10,labels=FALSE)
#Perform 10 fold cross validation
for(i in 1:10){
    #Segement your data by fold using the which() function 
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- nswshuffled[testIndexes, ]
    trainData <- nswshuffled[-testIndexes, ]
    #Use the test and train data partitions however you desire...
}
fit2 <- lm(EventPrice ~ AreaSize + Bedrooms + FirstAdvertisedEventPrice + LastAdvertisedEventPrice, data=trainData)
testData$result2 <- predict(fit2,testData)

print(testData$result2)