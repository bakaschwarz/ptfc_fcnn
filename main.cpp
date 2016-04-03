#include <ArgException.h>
#include "main.h"

/**
 * How does it work?
 * No args: Prints help
 * Otherwise: ONLY full args, not partial. So use with the GUI or you are on your own. At least for now.
 */
int main(int argc, const char** argv) {
    try
    {
        using namespace TCLAP;
        CmdLine cmd(
                "Please use the GUI Interface when possible.",
                ' ',
                "0.9"
        );
        UnlabeledValueArg<std::string> netPath(
                "neuralNetPath",
                "The path to the neural net file. This is both input AND output. If the file does not exist, it will be created.",
                true,
                "",
                "/path/to/neural.net"
        );
        UnlabeledValueArg<std::string> trainingPath(
                "trainingPath",
                "The path to the training file. For the format refer to the FCNN homepage.",
                true,
                "",
                "/path/to/training.dat"
        );

        cmd.add(trainingPath);
        cmd.add(netPath);
        cmd.parse(argc, argv);
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << "for arg" << e.argId() << std::endl;
    }
    return 0;
}

