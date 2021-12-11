import math
import time

def load_data(file_name, instance_cnt):
    try:
        f = open(file_name, 'r')
    except:
        raise FileNotFoundError(file_name)
    rows = [[] for i in range(instance_cnt)]
    for i in range(instance_cnt):
        rows[i] = [float(j) for j in f.readline().split()] 
    return rows

def leave_one_out_cross_validation(data, feature_set, feature_to_add):
    object_cnt = len(data)
    current_set = feature_set.copy()
    current_set.add(feature_to_add)
    object_correctly_classified = 0
    #print("----Current features: ", current_set)

    for i in range(object_cnt):
        object_to_classify = data[i][1:]
        object_to_classify_label = data[i][0]
        nn_distance = -1
        nn_location = 0
        nn_label = 0

        for j in range(object_cnt):
            distance = 0
            sum = 0
            if (i != j):
                #print("!!!i=%d and j=%d", i, j)
                for k in current_set:
                    sum = sum + (data[i][k] - data[j][k]) ** 2
                distance = math.sqrt(sum)
                if (nn_distance == -1):
                    nn_distance = distance
                elif (distance < nn_distance):
                    nn_distance = distance
                    nn_location = j
                    nn_label = data[j][0]
            #print("------On level %d i sum=%d and dist=%f and nn_dist=%f and nn_label=%f", i, sum, distance, nn_distance, nn_label)

        if (object_to_classify_label == nn_label):
            object_correctly_classified = object_correctly_classified + 1
            #print("----On level %d i classified=%d and object_to_classify_label=%f and nn_label=%d and cnt=%f" % (i, object_correctly_classified, object_to_classify_label, nn_label, object_cnt))

    accuracy = object_correctly_classified / object_cnt
    #print("----On level %d i nn_dist=%f and nn_loc=%d and accu=%f" % (i, nn_distance, nn_location, accuracy))
    return accuracy
    #print("Looping over i, at the %d location", i)
    #print("The %d object in the class %d, i, label_object_to_classify")



def forward_selection(data, feature_cnt):
    current_set_of_features = set()

    for i in range(1, feature_cnt+1):
        feature_to_add_at_this_level = 0
        best_accuracy_so_far = 0
        print("On the %d level of the search tree", i)
        for j in range(1, feature_cnt+1):
            if (j not in current_set_of_features):
                #print("--available features: ", current_set_of_features)
                accuracy = float(leave_one_out_cross_validation(data, current_set_of_features, j))

                if (accuracy > best_accuracy_so_far):
                    best_accuracy_so_far = accuracy
                    feature_to_add_at_this_level = j
                    #print("--j %d is higher...." % (feature_to_add_at_this_level))

                #print("--Considering adding %d feature with accuracy %f (best accu=%f)" %(j, accuracy, best_accuracy_so_far))

        current_set_of_features.add(feature_to_add_at_this_level)
        print("On level %d i added feature %s to current level with accuracy %f" % (i, current_set_of_features, best_accuracy_so_far))
        #print("On level %d i added feature %d to current level with accuracy %f" % (i, feature_to_add_at_this_level, best_accuracy_so_far))
        #print("#####Testing features: ", current_set_of_features,)



def backward_elemenation(data, feature_cnt):
    current_set_of_features = set(i for i in range(1, feature_cnt + 1))

    for i in range(1, feature_cnt+1):
        feature_to_remove_at_this_level = 0
        best_accuracy_so_far = 0
        print("On the %d level of the search tree", i)
        for j in range(1, feature_cnt+1):
            if (j in current_set_of_features):
                #print("--available features: ", current_set_of_features)
                accuracy = float(leave_one_out_cross_validation(data, current_set_of_features, j))

                if (accuracy > best_accuracy_so_far):
                    best_accuracy_so_far = accuracy
                    feature_to_remove_at_this_level = j
                    #print("--j %d is higher...." % (feature_to_add_at_this_level))

                #print("--Considering adding %d feature with accuracy %f (best accu=%f)" %(j, accuracy, best_accuracy_so_far))

        current_set_of_features.remove(feature_to_remove_at_this_level)
        print("On level %d i removed feature %s to current level with accuracy %f" % (i, current_set_of_features, best_accuracy_so_far))
        #print("On level %d i removed feature %d to current level with accuracy %f" % (i, feature_to_remove_at_this_level, best_accuracy_so_far))
        #print("#####Testing features: ", current_set_of_features,)



def main():
    file_name = input("Enter the name of the input data file: ")
    row_cnt = int(input("Enter the number of rows to read: "))
    data = load_data(file_name, row_cnt)

    algo = ""
    while (algo != "FS" and algo != "BE"):
        algo = input("""Type in the algorithm you want to run:
                       FS - Forward Selection
                       BE - Backward Elimination
                    \r""")

    feature_cnt = len(data[0]) - 1
    print("There are %d features with %d instances." % (feature_cnt, row_cnt))
    start_time = time.time()
    if (algo == "FS"):
        print("\t*** Starting Forward Selection... ***")
        forward_selection(data, feature_cnt)
    elif (algo == "BE"):
        print("BE")
        backward_elemenation(data, feature_cnt)
    else:
        print("Enter correct algorithm")
    end_time = time.time()
    print("Total compute time is %s seconds" % (end_time - start_time))

if __name__ == "__main__":
    main()