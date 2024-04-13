from chains.chain import review_chain 

context = "Legal stuff is cool"
question = "What is inmigration law"

response = review_chain.invoke({
    "context": context,
    "question": question
})

print(response)