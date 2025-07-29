import main

text_a = ['My name is John. I am a software engineer.',
          'I love programming and solving complex problems.',
          'In my free time, I enjoy reading books and hiking.',
          'I am passionate about learning new technologies and improving my skills.',
          'Collaboration and teamwork are important to me.']
text_b = ['My name is Alice. I am a data scientist.',
          'I enjoy analyzing data and building machine learning models.',
          'In my spare time, I like to cook and experiment with new recipes.',
          'I am enthusiastic about data visualization and storytelling with data.',
          'I believe in the power of data to drive decisions and innovation.']
steering_vector = main.calculate_steering_vector(text_a, text_b, batch_size=16)
print("Steering Vector:", steering_vector)
