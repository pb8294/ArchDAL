import os
import time
import pickle

class Checkpoint:

    def __init__(self, acc_list=None, loss_list=None, indices=None, state_dict=None, experiment_name=None, path=None):

        # If a path is supplied, load a checkpoint from there.
        if path is not None:

            if experiment_name is not None:
                self.load_checkpoint(path, experiment_name)
            else:
                raise ValueError("Checkpoint contains None value for experiment_name")

            return
        
        if loss_list is None:
            raise ValueError("Checkpoint contains None value for loss_list")

        if acc_list is None:
            raise ValueError("Checkpoint contains None value for acc_list")

        if indices is None:
            raise ValueError("Checkpoint contains None value for indices")

        if state_dict is None:
            raise ValueError("Checkpoint contains None value for state_dict")

        if experiment_name is None:
            raise ValueError("Checkpoint contains None value for experiment_name")

        self.acc_list = acc_list
        self.loss_list = loss_list
        self.indices = indices
        self.state_dict = state_dict
        self.experiment_name = experiment_name

    def __eq__(self, other):

        # Check if the accuracy lists are equal
        acc_lists_equal = self.acc_list == other.acc_list

        # Check if the indices are equal
        loss_lists_equal = self.loss_list == other.loss_list

        # Check if the indices are equal
        indices_equal = self.indices == other.indices

        # Check if the experiment names are equal
        experiment_names_equal = self.experiment_name == other.experiment_name

        return acc_lists_equal and indices_equal and experiment_names_equal

    def save_checkpoint(self, path, round):

        # Get current time to use in file timestamp
        timestamp = time.time()

        # Create the path supplied
        os.makedirs(path, exist_ok=True)

        # Name saved files using timestamp to add recency information
        save_path = os.path.join(path, F"c-rd_{str(round)}1")
        copy_save_path = os.path.join(path, F"c-rd_{str(round)}2")

        # Write this checkpoint to the first save location
        with open(save_path, 'wb') as save_file:
            pickle.dump(self, save_file)

        # Write this checkpoint to the second save location
        with open(copy_save_path, 'wb') as copy_save_file:
            pickle.dump(self, copy_save_file)

    def load_checkpoint(self, path, experiment_name):

        # Obtain a list of all files present at the path
        timestamp_save_no = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

        # If there are no such files, set values to None and return
        if len(timestamp_save_no) == 0:
            self.acc_list = None
            self.loss_list = None
            self.indices = None
            self.state_dict = None
            return

        # Sort the list of strings to get the most recent
        timestamp_save_no.sort(reverse=True)

        # Read in two files at a time, checking if they are equal to one another. 
        # If they are equal, then it means that the save operation finished correctly.
        # If they are not, then it means that the save operation failed (could not be 
        # done atomically). Repeat this action until no possible pair can exist.
        while len(timestamp_save_no) > 1:

            # Pop a most recent checkpoint copy
            first_file = timestamp_save_no.pop(0)

            # Keep popping until two copies with equal timestamps are present
            while True:
                
                second_file = timestamp_save_no.pop(0)
                
                # Timestamps match if the removal of the "1" or "2" results in equal numbers
                if (second_file[:-1]) == (first_file[:-1]):
                    break
                else:
                    first_file = second_file

                    # If there are no more checkpoints to examine, set to None and return
                    if len(timestamp_save_no) == 0:
                        self.acc_list = None
                        self.loss_list = None
                        self.indices = None
                        self.state_dict = None
                        return

            # Form the paths to the files
            load_path = os.path.join(path, first_file)
            copy_load_path = os.path.join(path, second_file)

            # Load the two checkpoints
            with open(load_path, 'rb') as load_file:
                checkpoint = pickle.load(load_file)

            with open(copy_load_path, 'rb') as copy_load_file:
                checkpoint_copy = pickle.load(copy_load_file)

            # Do not check this experiment if it is not the one we need to restore
            if checkpoint.experiment_name != experiment_name:
                continue

            # Check if they are equal
            if checkpoint == checkpoint_copy:

                # This checkpoint will suffice. Populate this checkpoint's fields 
                # with the selected checkpoint's fields.
                self.acc_list = checkpoint.acc_list
                self.loss_list = checkpoint.loss_list
                self.indices = checkpoint.indices
                self.state_dict = checkpoint.state_dict
                return

        # Instantiate None values in acc_list, indices, and model
        self.acc_list = None
        self.loss_list = None
        self.indices = None
        self.state_dict = None

    def get_saved_values(self):

        return (self.acc_list, self.loss_list, self.indices, self.state_dict)

def delete_checkpoints(checkpoint_directory, experiment_name):

    # Iteratively go through each checkpoint, deleting those whose experiment name matches.
    timestamp_save_no = [f for f in os.listdir(checkpoint_directory) if os.path.isfile(os.path.join(checkpoint_directory, f))]

    for file in timestamp_save_no:

        delete_file = False

        # Get file location
        file_path = os.path.join(checkpoint_directory, file)

        if not os.path.exists(file_path):
            continue

        # Unpickle the checkpoint and see if its experiment name matches
        with open(file_path, "rb") as load_file:

            checkpoint_copy = pickle.load(load_file)
            if checkpoint_copy.experiment_name == experiment_name:
                delete_file = True

        # Delete this file only if the experiment name matched
        if delete_file:
            os.remove(file_path)

#Logs
def write_logs(logs, save_directory, rd):
  file_path = save_directory + '/run_'+'.txt'
  with open(file_path, 'a') as f:
    f.write('---------------------\n')
    f.write('Round '+str(rd)+'\n')
    f.write('---------------------\n')
    for key, val in logs.items():
      if key == 'Training':
        f.write(str(key)+ '\n')
        for epoch in val:
          f.write(str(epoch)+'\n')       
      else:
        f.write(str(key) + ' - '+ str(val) +'\n')