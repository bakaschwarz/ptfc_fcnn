#include <ArgException.h>
#include "main.h"

int main(int argc, const char** argv) {
    try
    {
        //region TCLAP
        using namespace TCLAP;
        CmdLine cmd(
                "Please use the GUI Interface when possible.",
                ' ',
                "0.9"
        );
        UnlabeledValueArg<std::string> netPath(
                "neuralnet_path",
                "The path to the neural net file. This is both input AND output. If the file does not exist, it will be created.",
                true,
                "",
                "/path/to/neural.net"
        );
        UnlabeledValueArg<std::string> trainingPath(
                "training_path",
                "The path to the training file. Setting this will also start a training on the given net. For the format refer to the FCNN homepage.",
                true,
                "",
                "/path/to/training.dat"
        );
        ValueArg<std::string> printInfo(
                "",
                "print_info",
                "Using this will cause the program to give information about the specified file and prevents any training. Possible values: training, neuralnet",
                false,
                "",
                "training/neuralnet"
        );
        MultiArg<int> layers(
                "l",
                "layers",
                "Describes the number of layers for the network. Only has an effect if no network exists and the -n flag does point to a not existing file. You need to provide at least 3 values. First is the input, second a hidden layer and third the output layer.",
                false,
                "int int int ..."
        );
        SwitchArg test(
                "",
                "test",
                "Setting this flag will prevent any training and will use the given training file for testing the network instead. You will be provided with pairs of expected and actual results from the neural net."
        );
        ValueArg<float> error(
                "e",
                "desired_error_rate",
                "Use this to set the desired error rate. Default: 0.07f",
                false,
                0.07f,
                "float"
        );
        ValueArg<int> epoches(
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
        cmd.add(trainingPath);
        cmd.add(printInfo);
        cmd.add(netPath);
        cmd.parse(argc, argv);
        //endregion


    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << "for arg" << e.argId() << std::endl;
    }
    return 0;
}

