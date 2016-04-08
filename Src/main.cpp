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
        TCLAP::SwitchArg printInfo(
                "",
                "print_info",
                "Using this will cause the program to print information about the net and the training data. The information is JSON formatted for easy parsing."
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
        TCLAP::ValueArg<int> freq(
                "f",
                "frequency",
                "Sets the frequency for the reports. Default: 10",
                false,
                10,
                "int"
        );
        TCLAP::ValueArg<float> learnRate(
                "r",
                "learn_rate",
                "Sets the learning rate for this training session",
                false,
                0.5,
                "float"
        );
        cmd.add(learnRate);
        cmd.add(freq);
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
        bool newNet = false;
        if(fexists(neuralnetPath))
        {
            net.load(neuralnetPath);
        }
        else
        {
            net.construct(layers.getValue());
            net.set_name("This network was generated using ptfc_fann.");
            newNet = true;
        }

        if(fexists(datasetPath))
        {
            if(!dataset.load(datasetPath, true))
            {
                std::cerr << "Could not read dataset!" << std::endl;
                exit(-1);
            }
        }
        else
        {
            std::cerr << "No dataset found!" << std::endl; //TODO
            exit(-1);
        }
        //endregion

        if(printInfo.getValue())
        {
            //region Print information in JSON format
            std::cout << "{" << std::endl;
            std::cout << "\t\"NEURALNET\": {" << std::endl;
            std::cout << "\t\t\"DESCRIPTION\": \"" << net.get_name() << "\"," << std::endl;
            std::printf("\t\t\"NUMBERLAYERS\": %i,\n", net.no_layers());
            std::cout << "\t\t\"LAYERS\": [" << std::endl;
            for(int i = 1; i <= net.no_layers(); i++)
            {
                std::printf("\t\t\t{ \"%i\": %i}", i, net.no_neurons(i));
                if(i != net.no_layers())
                {
                    std::cout << ",";
                }
                std::cout << std::endl;
            }
            std::cout << "\t\t]" << std::endl;
            std::cout << "\t}," << std::endl;
            std::cout << "\t\"DATASETDESCRIPTION\": \"" << dataset.get_info() << "\"" << std::endl;
            std::cout << "}" << std::endl;
            //endregion
        }
        else if(test.getValue())
        {
            //region Test neural net
            if(!newNet)
            {
                std::cout << "Start testing the network..." << std::endl;
                float mse = net.mse(dataset);
                std::printf("Test finished!\nMSE: %f\n", mse);
            }
            else
            {
                std::cerr << "Can't test on a not existing network!" << std::endl;
                exit(-1);
            }
            //endregion
        }
        else
        {
            //region Configure net
            if(newNet)
            {
                for(int i = 2; i <= layers.getValue().size(); i++)
                {
                    net.set_act_f(i, fcnn::sigmoid);
                }
                net.rnd_weights(); //Randomize the weights
            }
            else
            {
                //TODO Is there even something to do?
            }
            //endregion

            //region Teach the net
            std::cout << "Starting training of the neural net..." << std::endl;
            fcnn::mlpnet_teach_bp(net, dataset, error.getValue(), epoches.getValue(), learnRate.getValue() ,freq.getValue());
            //endregion

            //region Save the net
            std::cout << "Finished training!\nNow saving..." << std::endl;
            net.save(neuralnetPath);
            //endregion
        }
    }
    catch (TCLAP::ArgException &e)
    {
        std::cerr << "error: " << e.error() << "for arg" << e.argId() << std::endl;
    }
    return 0;
}


inline bool fexists(const char* filename) {
    std::ifstream ifile(filename);
    bool ret = (bool) ifile;
    ifile.close();
    return ret;
}

