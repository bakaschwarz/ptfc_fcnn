add_library(fcnn fcnn dataset mat
        error export fcnncfg.h.in level1
        level1_impl.h level2 level3 level3
        matops matops_enable.h mlpnet
        mlpnet_prune mlpnet_teach
        rcarr report struct timing
        utils activation)
target_include_directories(fcnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})