import pandas as pd
import random
#test functions
def male_counts(sex_df):
    # Get the counts of each unique value in the "male" column
    male_counts = sex_df['male'].value_counts(dropna=False)

    # Print the counts
    print("Number of times 'male' is 0:", male_counts.get(0, 0))
    print("Number of times 'male' is 1:", male_counts.get(1, 0))
    print("Number of times 'male' is something else:", len(sex_df) - male_counts.get(0, 0) - male_counts.get(1, 0))
def check_if_both(train_df, column_name='same_text'):
    # Group by writer and check if both same_text=1 and same_text=0 are present
    writer_groups = train_df.groupby('writer')[column_name].nunique()

    # Filter writers that do not have both same_text=1 and same_text=0
    writers_missing_both = writer_groups[writer_groups != 2]

    if writers_missing_both.empty:
        print(f"All writers have both {column_name}=1 and {column_name}=0.")
    else:
        print(f"The following writers do not have {column_name}=1 and {column_name}=0")
        print(writers_missing_both)
def check_randomization(train_df):
    # Get the number of rows where train == 1
    train_1_count = train_df[train_df['train'] == 1].shape[0]

    # Calculate the fraction
    train_1_fraction = train_1_count / train_df.shape[0]

    print(f"Number of rows where train == 1: {train_1_count}")
    print(f"Fraction of rows where train == 1: {train_1_fraction:.2f}")
def check_grouping(train_df):
    # Group by writer and check if the train column has a constant value
    constant_train_check = train_df.groupby('writer')['train'].nunique()

    # Find writers where the train column is not constant
    non_constant_writers = constant_train_check[constant_train_check > 1]

    if non_constant_writers.empty:
        print("The train column is constant for all writers.")
    else:
        print("The train column is not constant for the following writers:")
        print(non_constant_writers)
def check_occurrences(train_df,count=4):
    # Count the occurrences of each unique writer value
    writer_counts = train_df['writer'].value_counts()

    # Check if all writers have exactly 4 occurrences
    if (writer_counts == count).all():
        print(f"Each unique writer value occurs on exactly {count} rows.")
    else:
        print(f"Some writers do not occur exactly {count} times.")
        print(writer_counts[writer_counts != 4])
def check_title_association(train_df):
    random_numbers = random.sample(range(1, 282*4+1), 10)
    for n in random_numbers:
        print(n)
        print(train_df['file_name'][n])
        print(train_df['writer'][n],train_df['isEng'][n], train_df['same_text'][n])
        print('-------------')
def check_sex_association(train_df,sex_df):
    random_numbers = random.sample(range(1, 283), 10)
    for n in random_numbers:
        print(n)
        print(train_df[train_df['writer'] == n][['writer','male']])
        print(sex_df[sex_df['writer'] == n][['writer','male']])
        print('-------------')
def check_if_seed(train_df):
    train_0_writers = train_df[train_df['train'] == 0]['writer'].unique().tolist()
    train_1_writers = train_df[train_df['train'] == 1]['writer'].unique().tolist()
    return train_0_writers, train_1_writers