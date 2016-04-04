add_library(tclap Arg.h ArgException.h ArgTraits.h CmdLine.h CmdLineInterface.h CmdLineOutput.h Constraint.h
        DocBookOutput.h HelpVisitor.h IgnoreRestVisitor.h MultiArg.h MultiSwitchArg.h OptionalUnlabeledTracker.h
        StandardTraits.h StdOutput.h SwitchArg.h UnlabeledMultiArg.h UnlabeledValueArg.h ValueArg.h ValuesConstraint.h
        VersionVisitor.h Visitor.h XorHandler.h ZshCompletionOutput.h)
set_target_properties(tclap PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(tclap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})