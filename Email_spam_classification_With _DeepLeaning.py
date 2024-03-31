import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('mail_data.csv')
data = df.where((pd.notnull(df)), '')

# Convert labels to integers
data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Split data
X = data['Message']
Y = data['Category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

# Vectorize text data
tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = tfidf_vectorizer.fit_transform(X_train)
X_test_features = tfidf_vectorizer.transform(X_test)

# Convert to PyTorch tensors
X_train_features = torch.tensor(X_train_features.toarray(), dtype=torch.float32)
X_test_features = torch.tensor(X_test_features.toarray(), dtype=torch.float32)
Y_train = torch.tensor(Y_train.astype('int').values, dtype=torch.long)
Y_test = torch.tensor(Y_test.astype('int').values, dtype=torch.long)

# Define a simple neural network model
class SpamHamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamHamClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 2)  # Output size is 2 for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialize the model
input_dim = X_train_features.shape[1]
model = SpamHamClassifier(input_dim)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Move model and data to GPU if available
model.to(device)
X_train_features, X_test_features = X_train_features.to(device), X_test_features.to(device)
Y_train, Y_test = Y_train.to(device), Y_test.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train_features), batch_size):
        optimizer.zero_grad()
        batch_X = X_train_features[i:i+batch_size]
        batch_Y = Y_train[i:i+batch_size]
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        optimizer.step()

    # Evaluate on training and validation set
    with torch.no_grad():
        model.eval()  # Set model to evaluation mode
        train_outputs = model(X_train_features)
        train_loss = criterion(train_outputs, Y_train)
        train_accuracy = accuracy_score(Y_train.cpu().numpy(), np.argmax(train_outputs.cpu().numpy(), axis=1))

        test_outputs = model(X_test_features)
        test_loss = criterion(test_outputs, Y_test)
        test_accuracy = accuracy_score(Y_test.cpu().numpy(), np.argmax(test_outputs.cpu().numpy(), axis=1))

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, '
            f'Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, '
            f'Test Accuracy: {test_accuracy:.4f}')

# Sample Input Message
input_message_mail = ["sometimes it’s easy to forget that good old SMS still exists. It may not be the spiffiest messaging technology out there, but the one great thing about SMS is that it's universal; you may now know whether someone is on Facebook Messenger or WhatsApp, but if you know their phone number, it's nearly certain they'll be able to receive an SMS message. What's even better is that the message technology is pretty universal, meaning you can even send a text from email."]

# Feature Extraction
input_data_features = tfidf_vectorizer.transform(input_message_mail)

# Convert features to a PyTorch Tensor and move to GPU
input_data_tensor = torch.tensor(input_data_features.toarray(), dtype=torch.float32).to(device)

# Switch model to evaluation mode
model.eval()

# Prediction
with torch.no_grad():
    outputs = model(input_data_tensor)
    prediction = torch.argmax(outputs, dim=1)  

# Output Formatting
print("Result score of prediction =", prediction.item())  # Get integer value

if prediction.item() == 1: 
    print("Ham mail")
else:
    print("Spam mail")