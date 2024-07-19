# Sample list of accuracy values
accuracy = [92.60,94.14,92.60,98.01,98.01,91.05,98.92,96.86,94.08,94.14,86.15,91.05,86.15,72.06,83.29,
            72.66,56.68,96.86,92.60,80.65,91.10,86.15,97.62,86.50,75.15,98.01,91.05,91.10,94.08]

# Calculate the average accuracy
average_accuracy = sum(accuracy) / len(accuracy)

# Print the result
print(f"Average Accuracy : {average_accuracy:.2f}%")