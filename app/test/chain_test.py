from chains.chain import chain 

context = "Legal stuff is cool"
question = "What is inmigration law"

response = chain.invoke({
    "context": context,
    "question": question
})

print(response)