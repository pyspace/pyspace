""" Generates an arff file with the classes A and B, 1000 features and 500 instances """
if __name__ == "__main__":
    import random

    features = 1000
    instances = 500

    arff_file = open("random.arff", "w")

    arff_file.write("@relation 'random'\n")
    for i in range(features):
      arff_file.write("@attribute feature_%s real\n" % i)

    arff_file.write("@attribute class {A, B}\n")
    arff_file.write("@data\n")

    for instance in range(instances):
      label = random.choice(["'A'","'B'"])
      for feature in range(features):
        arff_file.write("%s," % random.random())
      arff_file.write("%s\n" % label)



