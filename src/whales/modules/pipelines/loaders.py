from glob import glob

from whales.modules.module import Module


class PipelineLoaders(Module):
    def load_performance_indicators(self):
        pass

    def load_input_data(self):
        """Add instructions to load the input data. Also, use glob where stars are detected"""
        real_input_data = []
        for elem in self.parameters["input_data"]:
            if "file_name" in elem:
                if "*" in elem["file_name"]:
                    # Glob detected
                    file_names = glob(elem["file_name"])
                    data_files = [elem["data_file"]] * len(file_names)
                    formatters = [elem["formatter"]] * len(file_names)
                    new_elems = [{
                        "file_name": a,
                        "data_file": b,
                        "formatter": c,
                    } for a, b, c in zip(file_names, data_files, formatters)]
                    real_input_data += new_elems
                else:
                    real_input_data.append(elem)
        data_set_type = self.parameters.get("data_set_type", "files_fold")
        self.add_instruction("build_data_set",
                             {
                                 "input_data": real_input_data,
                                 "data_set_type": data_set_type,
                                 "sliding_windows": {
                                     "overlap": 0.3,
                                     "sliding_window_width": "60s",
                                 }
                             })

    def load_labels(self):
        """Add instructions to load the labels. Also, use glob where stars are detected"""
        real_labels = []
        for elem in self.parameters["input_labels"]:
            if "labels_file" in elem:
                if "*" in elem["labels_file"]:
                    labels_files = glob(elem["labels_file"])
                    labels_formatters = [elem["labels_formatter"]] * len(labels_files)
                    new_elems = [{
                        "labels_file": a,
                        "labels_formatter": b,
                    } for a, b in zip(labels_files, labels_formatters)]
                    real_labels += new_elems
                else:
                    real_labels.append({
                        elem["labels_file"],
                        elem["labels_formatter"]
                    })
        self.add_instruction("set_labels", {"input_labels": real_labels})

    def load_features_extractors(self):
        """Add instructions to load the features extractors"""
        real_labels = []
        for elem in self.parameters["features_extractors"]:
            if "labels_file" in elem:
                if "*" in elem["labels_file"]:
                    labels_files = glob(elem["labels_file"])
                    labels_formatters = [elem["labels_formatter"]] * len(labels_files)
                    new_elems = [{
                        "labels_file": a,
                        "labels_formatter": b,
                    } for a, b in zip(labels_files, labels_formatters)]
                    real_labels += new_elems
                else:
                    real_labels.append({
                        elem["labels_file"],
                        elem["labels_formatter"]
                    })
        self.add_instruction("set_labels", {"input_labels": real_labels})