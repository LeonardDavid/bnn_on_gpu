#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <tuple>

#include "cuda_net.h"
#include "net.hpp"
#include "netW.hpp"

using namespace std;

std::tuple<float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float,float>  
predict_NeuralNet(unsigned char x[][32][32][3], float * pred) { 

  float kernel_time = 0;
  float sum_cpu = 0;
  float sum_gpu = 0;
  /* Layer 1 CPU */
  // // Layer 1: Conv @ cpp.NHWC {% else %} /{% if pads == [0, 0, 0, 0] %}
  // auto start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 32; h++) {
  //     for (int w = 0; w < 32; w++) {
  //       for (int m = 0; m < 128; m++) {
  //         cuda_layer_1_output[index4D(b,h,w,m,32,32,128)] = layer_1_bias[m]; // layer_1_output[b][h][w][m] / cuda_layer_1_output[index4D(b,h,w,m,32,32,128)]
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 32) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 32) {
  //               for (int c = 0; c < 3; c++) {
  //                 for (int m = 0; m < 128; m++) {
  //                   cuda_layer_1_output[index4D(b,h,w,m,32,32,128)] += layer_1_weight[kH][kW][c][m] * x[b][iH][iW][c]; // layer_1_output[b][h][w][m] / cuda_layer_1_output[index4D(b,h,w,m,32,32,128)] // x[b][iH][iW][c] / x[index4D(b,iH,iW,c,32,32,3)]
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // auto end = std::chrono::high_resolution_clock::now();
  // auto l1_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l1_kernel_time = 0;

  /* Layer 1 GPU */
  auto start = std::chrono::high_resolution_clock::now();
  kernel_time += layer1_conv(x, cuda_layer_1_output);
  auto end = std::chrono::high_resolution_clock::now();
  auto l1_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l1_kernel_time = kernel_time;
  l1_time -= l1_kernel_time*1000000.0f; // ms->ns

  // // checksum L1 = 5720315.5
  // ofstream gg1("layer1/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //     sum_gpu = 0;
  //     for(int i=b*32*32*128;i<(b+1)*32*32*128;i++){
  //         sum_gpu += cuda_layer_1_output[i];
  //         gg1<<cuda_layer_1_output[i]<<" ";  
  //     }
  //     cout<<fixed<<"layer 1(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 2 CPU */
  // Layer 2: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
  start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 32; h++) {
      for (int w = 0; w < 32; w++) {
        for (int c = 0; c < 128; c++) {
          if (cuda_layer_1_output[index4D(b,h,w,c,32,32,128)] > layer_2_threshold[c]) { // layer_1_output[b][h][w][c] , cuda_layer_1_output[index4D(b,h,w,c,32,32,128)]
            layer_2_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_2_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
  }

  // unsigned long long *cuda_layer_2_output = (unsigned long long *) layer_2_output;
  // ^ direct pointer assignment leads to segmentation fault ^
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 32; h++) {
      for (int w = 0; w < 32; w++) {
        for (int c = 0; c < 128; c++) {
          cuda_layer_2_output[index4D(b,h,w,c,32,32,128)] = layer_2_output[b][h][w][c];
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l2_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  

  /* Layer 3 CPU */
  // // Layer 3: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 32; h++) {
  //     for (int w = 0; w < 32; w++) {
  //       for (int m = 0; m < 128; m++) {
  //         cuda_layer_3_output[index4D(b,h,w,m,32,32,128)] = layer_3_bias[m]; // layer_3_output[b][h][w][m] / cuda_layer_3_output[index4D(b,h,w,m,32,32,128)]
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 32) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 32) {
  //               for (int m = 0; m < 128; m++) {
  //                 for (int c = 0; c < 2; c++) {
  //                   cuda_layer_3_output[index4D(b,h,w,m,32,32,128)] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_3_weight[kH][kW][m][c] ^ cuda_layer_2_output[index4D(b,iH,iW,c,32,32,128)])) - 64; // layer_2_output[b][iH][iW][c] / cuda_layer_2_output[index4D(b,iH,iW,c,32,32,2)]
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l3_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l3_kernel_time = 0;    

  /* Layer 3 GPU */ 
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer3_conv(cuda_layer_2_output, cuda_layer_3_output);
  end = std::chrono::high_resolution_clock::now();  
  auto l3_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l3_kernel_time = kernel_time-l1_kernel_time;
  l3_time -= l3_kernel_time*1000000.0f; // ms->ns

  // // checksum L3 = -2335755.75
  // ofstream gg3("layer3/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*32*32*128;i<(b+1)*32*32*128;i++){
  //       sum_gpu += cuda_layer_3_output[i];
  //       gg3<<cuda_layer_3_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 3(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 4 CPU */
  // // Layer 4: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 16; h++) {
  //     for (int w = 0; w < 16; w++) {
  //       for (int c = 0; c < 128; c++) {
  //         cuda_layer_4_output[index4D(b,h,w,c,16,16,128)] = std::numeric_limits<float>::lowest(); // layer_4_output[b][h][w][m] / cuda_layer_4_output[index4D(b,h,w,c,16,16,128)]
  //       }
  //       for (int kH = 0; kH < 2; kH++) {
  //         for (int kW = 0; kW < 2; kW++) {
  //           for (int c = 0; c < 128; c++) {
  //             cuda_layer_4_output[index4D(b,h,w,c,16,16,128)] = std::max(cuda_layer_3_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,32,32,128)], cuda_layer_4_output[index4D(b,h,w,c,16,16,128)]); // layer_3_output[b][h * 2 + kH][w * 2 + kW][c] / cuda_layer_3_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,32,32,128)]
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  // float l4_kernel_time = 0;      

  /* Layer 4 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer4_maxpool(cuda_layer_3_output, cuda_layer_4_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l4_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l4_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time);
  l4_time -= l4_kernel_time*1000000.0f; // ms->ns

  // // checksum L4 = 1633936.0
  // ofstream gg4("layer4/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*16*16*128;i<(b+1)*16*16*128;i++){
  //       sum_gpu += cuda_layer_4_output[i];
  //       gg4<<cuda_layer_4_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 4(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 5 CPU */
  // Layer 5: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
  start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int c = 0; c < 128; c++) {
          if (cuda_layer_4_output[index4D(b,h,w,c,16,16,128)] > layer_5_threshold[c]) { // layer_4_output[b][h][w][c] / cuda_layer_4_output[index4D(b,h,w,c,16,16,128)]
            layer_5_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_5_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
  }

  // unsigned long long *cuda_layer_5_output = (unsigned long long *) layer_5_output;
  // ^ direct pointer assignment leads to segmentation fault ^
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int c = 0; c < 128; c++) {
          cuda_layer_5_output[index4D(b,h,w,c,16,16,128)] = layer_5_output[b][h][w][c];
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l5_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());    

  /* Layer 6 CPU */
  // // Layer 6: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 16; h++) {
  //     for (int w = 0; w < 16; w++) {
  //       for (int m = 0; m < 256; m++) {
  //         cuda_layer_6_output[index4D(b,h,w,m,16,16,256)] = layer_6_bias[m]; // layer_6_output[b][h][w][m] / cuda_layer_6_output[index4D(b,h,w,m,16,16,256)]
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 16) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 16) {
  //               for (int m = 0; m < 256; m++) {
  //                 for (int c = 0; c < 2; c++) {
  //                   cuda_layer_6_output[index4D(b,h,w,m,16,16,256)] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_6_weight[kH][kW][m][c] ^ cuda_layer_5_output[index4D(b,iH,iW,c,16,16,128)])) - 64; // layer_5_output[b][iH][iW][c] / cuda_layer_5_output[index4D(b,iH,iW,c,16,16,128)]
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l6_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l6_kernel_time = 0;     

  /* Layer 6 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer6_conv(cuda_layer_5_output, cuda_layer_6_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l6_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l6_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time);
  l6_time -= l6_kernel_time*1000000.0f; // ms->ns

  // // checksum L6 = -20699.617188
  // ofstream gg6("layer6/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*16*16*256;i<(b+1)*16*16*256;i++){
  //       sum_gpu += cuda_layer_6_output[i];
  //       gg6<<cuda_layer_6_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 6(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 7 CPU */
  // Layer 7: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
  start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int c = 0; c < 256; c++) {
          if (cuda_layer_6_output[index4D(b,h,w,c,16,16,256)] > layer_7_threshold[c]) { // layer_6_output[b][h][w][c] / cuda_layer_6_output[index4D(b,h,w,c,16,16,256)]
            layer_7_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_7_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
  }

  // unsigned long long *cuda_layer_7_output = (unsigned long long *) layer_7_output;
  // ^ direct pointer assignment leads to segmentation fault ^
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 16; h++) {
      for (int w = 0; w < 16; w++) {
        for (int c = 0; c < 256; c++) {
          cuda_layer_7_output[index4D(b,h,w,c,16,16,256)] = layer_7_output[b][h][w][c];
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l7_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());   

  /* Layer 8 CPU */
  // // Layer 8: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 16; h++) {
  //     for (int w = 0; w < 16; w++) {
  //       for (int m = 0; m < 256; m++) {
  //         cuda_layer_8_output[index4D(b,h,w,m,16,16,256)] = layer_8_bias[m]; // layer_8_output[b][h][w][m] / 
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 16) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 16) {
  //               for (int m = 0; m < 256; m++) {
  //                 for (int c = 0; c < 4; c++) {
  //                   cuda_layer_8_output[index4D(b,h,w,m,16,16,256)] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_8_weight[kH][kW][m][c] ^ cuda_layer_7_output[index4D(b,iH,iW,c,16,16,256)])) - 64; // layer_7_output[b][iH][iW][c] / cuda_layer_7_output[index4D(b,iH,iW,c,16,16,256)]
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l8_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l8_kernel_time = 0;    

  /* Layer 8 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer8_conv(cuda_layer_7_output, cuda_layer_8_output);
  end = std::chrono::high_resolution_clock::now();  
  auto l8_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  float l8_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time+l6_kernel_time);
  l8_time -= l8_kernel_time*1000000.0f; // ms->ns

  // // checksum L8 = -225414.96875
  // ofstream gg8("layer8/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*16*16*256;i<(b+1)*16*16*256;i++){
  //       sum_gpu += cuda_layer_8_output[i];
  //       gg8<<cuda_layer_8_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 8(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 9 CPU */
  // // Layer 9: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 8; h++) {
  //     for (int w = 0; w < 8; w++) {
  //       for (int c = 0; c < 256; c++) {
  //         cuda_layer_9_output[index4D(b,h,w,c,8,8,256)] = std::numeric_limits<float>::lowest();
  //       }
  //       for (int kH = 0; kH < 2; kH++) {
  //         for (int kW = 0; kW < 2; kW++) {
  //           for (int c = 0; c < 256; c++) {
  //             cuda_layer_9_output[index4D(b,h,w,c,8,8,256)] = std::max(cuda_layer_8_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,16,16,256)], cuda_layer_9_output[index4D(b,h,w,c,8,8,256)]); // layer_8_output[b][h * 2 + kH][w * 2 + kW][c] / cuda_layer_8_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,16,16,256)]
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l9_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l9_kernel_time = 0;    

  /* Layer 9 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer9_maxpool(cuda_layer_8_output, cuda_layer_9_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l9_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l9_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time+l6_kernel_time+l8_kernel_time);
  l9_time -= l9_kernel_time*1000000.0f; // ms->ns

  // // checksum L9 = 2192928.0
  // ofstream gg9("layer9/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*8*8*256;i<(b+1)*8*8*256;i++){
  //       sum_gpu += cuda_layer_9_output[i];
  //       gg9<<cuda_layer_9_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 9(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 10 CPU */
  // Layer 10: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
  start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int c = 0; c < 256; c++) {
          if (cuda_layer_9_output[index4D(b,h,w,c,8,8,256)] > layer_10_threshold[c]) { // layer_9_output[b][h][w][c] / cuda_layer_9_output[index4D(b,h,w,c,8,8,256)]
            layer_10_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_10_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
  }

  // unsigned long long *cuda_layer_10_output = (unsigned long long *) layer_10_output;
  //  ^ direct pointer assignment leads to segmentation fault ^
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int c = 0; c < 256; c++) {
          cuda_layer_10_output[index4D(b,h,w,c,8,8,256)] = layer_10_output[b][h][w][c];
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l10_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

  /* Layer 11 CPU */
  // // Layer 11: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 8; h++) {
  //     for (int w = 0; w < 8; w++) {
  //       for (int m = 0; m < 512; m++) {
  //         cuda_layer_11_output[index4D(b,h,w,m,8,8,512)] = layer_11_bias[m]; // layer_11_output[b][h][w][m] / cuda_layer_11_output[index4D(b,h,w,m,8,8,512)]
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 8) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 8) {
  //               for (int m = 0; m < 512; m++) {
  //                 for (int c = 0; c < 4; c++) {
  //                   cuda_layer_11_output[index4D(b,h,w,m,8,8,512)] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_11_weight[kH][kW][m][c] ^ cuda_layer_10_output[index4D(b,iH,iW,c,8,8,256)])) - 64; // layer_10_output[b][iH][iW][c] / cuda_layer_10_output[index4D(b,iH,iW,c,8,8,256)]
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l11_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l11_kernel_time = 0;     

  /* Layer 11 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer11_conv(cuda_layer_10_output, cuda_layer_11_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l11_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l11_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time+l6_kernel_time+l8_kernel_time+l9_kernel_time);
  l11_time -= l11_kernel_time*1000000.0f; // ms->ns

  // // checksum L11 = 38519.339844
  // ofstream gg11("layer11/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*8*8*512;i<(b+1)*8*8*512;i++){
  //       sum_gpu += cuda_layer_11_output[i];
  //       gg11<<cuda_layer_11_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 11(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 12 CPU */
  // Layer 12: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
  start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int c = 0; c < 512; c++) {
          if (cuda_layer_11_output[index4D(b,h,w,c,8,8,512)] >layer_12_threshold[c]) { // layer_11_output[b][h][w][c] / cuda_layer_11_output[index4D(b,h,w,c,8,8,512)]
            layer_12_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_12_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
  }

  // unsigned long long *cuda_layer_12_output = (unsigned long long *) layer_12_output;
  //  ^ direct pointer assignment leads to segmentation fault ^
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 8; h++) {
      for (int w = 0; w < 8; w++) {
        for (int c = 0; c < 512; c++) {
          cuda_layer_12_output[index4D(b,h,w,c,8,8,512)] = layer_12_output[b][h][w][c];
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l12_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());

  /* Layer 13 CPU */
  // // Layer 13: Conv @ cpp.binary {% else %} /{% if layer.pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 8; h++) {
  //     for (int w = 0; w < 8; w++) {
  //       for (int m = 0; m < 512; m++) {
  //         cuda_layer_13_output[index4D(b,h,w,m,8,8,512)] = layer_13_bias[m]; // layer_13_output[b][h][w][m] / cuda_layer_13_output[index4D(b,h,w,m,8,8,512)]
  //       }
  //       for (int kH = 0; kH < 3; kH++) {
  //         int iH = h * 1 + kH - 1;
  //         if (iH >= 0 && iH < 8) {
  //           for (int kW = 0; kW < 3; kW++) {
  //             int iW = w * 1 + kW - 1;
  //             if (iW >= 0 && iW < 8) {
  //               for (int m = 0; m < 512; m++) {
  //                 for (int c = 0; c < 8; c++) {
  //                   cuda_layer_13_output[index4D(b,h,w,m,8,8,512)] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_13_weight[kH][kW][m][c] ^ cuda_layer_12_output[index4D(b,iH,iW,c,8,8,512)])) - 64; // layer_12_output[b][iH][iW][c] / cuda_layer_12_output[index4D(b,iH,iW,c,8,8,512)]
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l13_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l13_kernel_time = 0;      

  /* Layer 13 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer13_conv(cuda_layer_12_output, cuda_layer_13_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l13_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l13_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time+l6_kernel_time+l8_kernel_time+l9_kernel_time+l11_kernel_time);
  l13_time -= l13_kernel_time*1000000.0f; // ms->ns

  // // checksum L13 = -125208.054688
  // ofstream gg13("layer13/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*8*8*512;i<(b+1)*8*8*512;i++){
  //       sum_gpu += cuda_layer_13_output[i];
  //       gg13<<cuda_layer_13_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 13(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 14 CPU */ 
  // // Layer 14: MaxPool @ cpp.NHWC {% if pads == [0, 0, 0, 0] %}
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int h = 0; h < 4; h++) {
  //     for (int w = 0; w < 4; w++) {
  //       for (int c = 0; c < 512; c++) {
  //         cuda_layer_14_output[index4D(b,h,w,c,4,4,512)] = std::numeric_limits<float>::lowest();
  //       }
  //       for (int kH = 0; kH < 2; kH++) {
  //         for (int kW = 0; kW < 2; kW++) {
  //           for (int c = 0; c < 512; c++) {
  //             cuda_layer_14_output[index4D(b,h,w,c,4,4,512)] = std::max(cuda_layer_13_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,8,8,512)], cuda_layer_14_output[index4D(b,h,w,c,4,4,512)]); // layer_13_output[b][h * 2 + kH][w * 2 + kW][c] / cuda_layer_13_output[index4D(b,(h * 2 + kH),(w * 2 + kW),c,8,8,512)]
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l14_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l14_kernel_time = 0;      

  /* Layer 14 GPU */ 
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer14_maxpool(cuda_layer_13_output, cuda_layer_14_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l14_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l14_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time+l6_kernel_time+l8_kernel_time+l9_kernel_time+l11_kernel_time+l13_kernel_time);
  l14_time -= l14_kernel_time*1000000.0f; // ms->ns

  // // checksum L14 = 1373773.625
  // ofstream gg14("layer14/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*4*4*512;i<(b+1)*4*4*512;i++){
  //       sum_gpu += cuda_layer_14_output[i];
  //       gg14<<cuda_layer_14_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 14(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 15 CPU */
  // Layer 15: Step @ cpp.binary {% if layer.output_shape|length > 2 %}
  start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int h = 0; h < 4; h++) {
      for (int w = 0; w < 4; w++) {
        for (int c = 0; c < 512; c++) {
          if (cuda_layer_14_output[index4D(b,h,w,c,4,4,512)] > layer_15_threshold[c]) { // layer_14_output[b][h][w][c] / cuda_layer_14_output[index4D(b,h,w,c,4,4,512)]
            layer_15_output[b][h][w][c / 64] |= (1ULL << (63 - c % 64));
          } else {
            layer_15_output[b][h][w][c / 64] &= ~(1ULL << (63 - c % 64));
          }
        }
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l15_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());   

  // unsigned long long *cuda_layer_15_output = (unsigned long long *) layer_15_output;

  // Layer 16: Flatten @ cpp.NHWC:reshape.j2 
  unsigned long long *cuda_layer_16_output = (unsigned long long *) layer_15_output;

  /* Layer 17 CPU */
  // // Layer 17: Gemm @ cpp.binary
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int d = 0; d < 1024; d++) {
  //     cuda_layer_17_output[b*1024 + d] = layer_17_bias[d];
  //   }
  //   for (int d = 0; d < 1024; d++) {
  //     for (int i = 0; i < 128; i++) {
  //       cuda_layer_17_output[b*1024 + d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_17_weight[d][i] ^ cuda_layer_16_output[b*128 + i])) - 64;
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l17_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l17_kernel_time = 0;     

  /* Layer 17 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer17_gemm(cuda_layer_16_output, cuda_layer_17_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l17_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l17_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time+l6_kernel_time+l8_kernel_time+l9_kernel_time+l11_kernel_time+l13_kernel_time+l14_kernel_time);
  l17_time -= l17_kernel_time*1000000.0f; // ms->ns

  // // checksum L17 = 10874.058594
  // ofstream gg17("layer17/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*1024;i<(b+1)*1024;i++){
  //       sum_gpu += cuda_layer_17_output[i];
  //       gg17<<cuda_layer_17_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 17(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  /* Layer 18 CPU */
  // Layer 18: Step @ cpp.binary {% else %} /{% if layer.output_shape|length > 2 %}
  start = std::chrono::high_resolution_clock::now();
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int d = 0; d < 1024; d++) {
      if (cuda_layer_17_output[b*1024 + d] >layer_18_threshold[d]) { // layer_17_output[b][d] / cuda_layer_17_output[b*1024 + d]
        layer_18_output[b][d / 64] |= (1ULL << (63 - d % 64));
      } else {
        layer_18_output[b][d / 64] &= ~(1ULL << (63 - d % 64));
      }
    }
  }
  end = std::chrono::high_resolution_clock::now();
  auto l18_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());   

  unsigned long long *cuda_layer_18_output = (unsigned long long *) layer_18_output;

  /* Layer 19 CPU */
  // // Layer 19: Gemm @ cpp.binary
  // start = std::chrono::high_resolution_clock::now();
  // for (int b = 0; b < BATCH_SIZE; b++){
  //   for (int d = 0; d < 10; d++) {
  //     cuda_layer_19_output[b*10 + d] = layer_19_bias[d];
  //   }
  //   for (int d = 0; d < 10; d++) {
  //     for (int i = 0; i < 16; i++) {
  //       cuda_layer_19_output[b*10 + d] += 2 * __builtin_popcountll((unsigned long long)~(unsigned long long)(layer_19_weight[d][i] ^ cuda_layer_18_output[b*16+i])) - 64; // layer_18_output[b][i] / cuda_layer_18_output[b*16+i]
  //     }
  //   }
  // }
  // end = std::chrono::high_resolution_clock::now();
  // auto l19_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
  // float l19_kernel_time = 0;     

  /* Layer 19 GPU */
  start = std::chrono::high_resolution_clock::now();
  kernel_time += layer19_gemm(cuda_layer_18_output, cuda_layer_19_output);
  end = std::chrono::high_resolution_clock::now();    
  auto l19_time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());  
  float l19_kernel_time = kernel_time-(l1_kernel_time+l3_kernel_time+l4_kernel_time+l6_kernel_time+l8_kernel_time+l9_kernel_time+l11_kernel_time+l13_kernel_time+l14_kernel_time+l17_kernel_time);
  l19_time -= l19_kernel_time*1000000.0f; // ms->ns

  // // checksum L19 = 16.014023
  // ofstream gg19("layer19/par.out");
  // for(int b=0;b<BATCH_SIZE;b++){
  //   sum_gpu = 0;
  //   for(int i=b*10;i<(b+1)*10;i++){
  //       sum_gpu += cuda_layer_19_output[i];
  //       gg19<<cuda_layer_19_output[i]<<" ";  
  //   }
  //   cout<<fixed<<"layer 19(GPU): batch "<<b<<": "<<sum_gpu<<endl;
  // }
  // cout<<endl;
  
  for (int b = 0; b < BATCH_SIZE; b++){
    for (int i = 0; i < 10; i++) {
      pred[b*10 + i] += cuda_layer_19_output[b*10 + i]; // layer_19_output[b][i] / cuda_layer_19_output[b*10 + i]
    }
  }

  return make_tuple(kernel_time, l1_time, l2_time, l3_time, l4_time, l5_time, l6_time, l7_time, l8_time, l9_time, 
    l10_time, l11_time, l12_time, l13_time, l14_time, l15_time, l17_time, l18_time, l19_time,
    l1_kernel_time, l3_kernel_time, l4_kernel_time, l6_kernel_time, l8_kernel_time, l9_kernel_time, 
    l11_kernel_time, l13_kernel_time, l14_kernel_time, l17_kernel_time, l19_kernel_time);

}
