#include "main.h"

int main(int argc, const char** argv) {
    try
    {
        //region TCLAP
        TCLAP::CmdLine cmd(
                "Please use the GUI Interface when possible.",
                ' ',
                "0.9"
        );
        TCLAP::ValueArg<std::string> netPath(
                "n",
                "neuralnet_path",
                "The path to the neural net file. This is both input AND output. If the file does not exist, it will be created. WARNING: Always use the full path! Not ~/file",
                true,
                "",
                "/path/to/neural.net"
        );
        TCLAP::ValueArg<std::string> trainingPath(
                "t",
                "training_path",
                "The path to the training file. Setting this will also start a training on the given net. For the format refer to the FCNN homepage.",
                true,
                "",
                "/path/to/training.dat"
        );
        TCLAP::ValueArg<std::string> printInfo(
                "",
                "print_info",
                "Using this will cause the program to give information about the specified file and prevents any training. Possible values: training, neuralnet",
                false,
                "",
                "training/neuralnet"
        );
        TCLAP::MultiArg<int> layers(
                "l",
                "layer",
                "Adds a layer to the network. You need at least 3 of these flags. One for the input, hidden and output layer. The order is important.",
                false,
                "int"
        );
        TCLAP::SwitchArg test(
                "",
                "test",
                "Setting this flag will prevent any training and will use the given training file for testing the network instead. You will be provided with pairs of expected and actual results from the neural net."
        );
        TCLAP::ValueArg<float> error(
                "e",
                "desired_error_rate",
                "Use this to set the desired error rate. Default: 0.07f",
                false,
                0.07f,
                "float"
        );
        TCLAP::ValueArg<int> epoches(
                "m",
                "max_epoches",
                "How many epoches of training are acceptable? Default: 5000",
                false,
                5000,
                "int"
        );
        cmd.add(epoches);
        cmd.add(error);
        cmd.add(test);
        cmd.add(layers);
        cmd.add(printInfo);
        cmd.add(trainingPath);
        cmd.add(netPath);
        cmd.parse(argc, argv);
        //endregion

        //region Load net and dataset
        fcnn::MLPNet<float> net;
        fcnn::Dataset<float> dataset;
        const char* neuralnetPath = netPath.getValue().c_str();
        const char* datasetPath = trainingPath.getValue().c_str();

        if(fexists(neuralnetPath))
        {
            std::cout << "Net exists!" << std::endl; //TODO
            net.load(neuralnetPath);
        }
        else
        {
            net.construct(layers.getValue());
        }

        if(fexists(datasetPath))
        {
            std::cout << "Dataset exists" << std::endl; //TODO
            dataset.load(datasetPath);
        }
        else
        {
            std::cerr << "No dataset found!" << std::endl;
            exit(-1);
        }
        //endregion
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << "for arg" << e.argId() << std::endl;
    }
    return 0;
}


bool fexists(const char* filename) {
    std::ifstream ifile(filename);
    return (bool) ifile;
}

