import os

class Directory:
    def get_all_directories(self, root_dir):
        assert os.path.isdir(root_dir), '{} is not a valid directory'.format(root_dir)
        return sorted( [os.path.join(root_dir, _dir) for _dir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, _dir))] )

    def get_all_files(self, root_dir):
        assert os.path.isdir(root_dir), '{} is not a valid directory'.format(root_dir)
        return sorted( [os.path.join(root_dir, _file) for _file in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, _file))] )

    def get_all_items(self, root_dir):
        assert os.path.isdir(root_dir), '{} is not a valid directory'.format(root_dir)
        return sorted( [os.path.join(root_dir, _file) for _file in os.listdir(root_dir)] )

    def get_all_files_of_type(self, root_dir, type):
        assert os.path.isdir(root_dir), '{} is not a valid directory'.format(root_dir)
        return sorted( [os.path.join(root_dir, _file) for _file in os.listdir(root_dir) if _file.endswith(type) and os.path.isfile(os.path.join(root_dir, _file))] )

    def is_file(self, entry):
        return os.path.isfile(entry)

    def is_dir(self, entry):
        return os.path.isdir(entry)

    def mkdir_if_not_exist(self, dir):
        if not self.is_dir(dir):
            os.makedirs(dir)