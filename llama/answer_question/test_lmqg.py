from lmqg import TransformersQG

model = TransformersQG(language="en")
context = "William Turner was an English painter who specialised in watercolour landscapes. He is often known " \
          "as William Turner of Oxford or just Turner of Oxford to distinguish him from his contemporary, " \
          "J. M. W. Turner. Many of Turner's paintings depicted the countryside around Oxford. One of his " \
          "best known pictures is a view of the city of Oxford from Hinksey Hill."
qa = model.generate_qa(context)