/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "arm_math_types.h"
#include "eth.h"
#include "fdcan.h"
#include "ltdc.h"
#include "quadspi.h"
#include "rtc.h"
#include "sai.h"
#include "usart.h"
#include "usb_otg.h"
#include "gpio.h"
#include "fmc.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
#include <stdio.h>
#include <string.h>
#include "lif_neuron_gen.h"
#include "test.h"


/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#ifndef HSEM_ID_0
#define HSEM_ID_0 (0U) /* HW semaphore 0*/
#endif

#define TEST_OUTPUT 20
#define NUM_STEPS 256 //number of timesteps to simulate
#define INPUTS 12
#define OUTPUTS 7

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
/* accumulate_output holds integer spike counts for each output neuron */
int32_t accumulate_output[OUTPUTS];


/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
void PeriphCommonClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* Configure the peripherals common clocks */
  PeriphCommonClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ADC1_Init();
  MX_ADC2_Init();
  MX_ADC3_Init();
  MX_ETH_Init();
  MX_FDCAN1_Init();
  MX_FDCAN2_Init();
  MX_FMC_Init();
  MX_LTDC_Init();
  MX_QUADSPI_Init();
  MX_RTC_Init();
  MX_SAI2_Init();
  MX_USART1_UART_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  /* USER CODE BEGIN 2 */

  float32_t result = 0;

  float32_t dsp_test()
{
    float32_t a[4] = {1,2,3,4};
    float32_t b[4] = {5,6,7,8};
    float32_t result;
    arm_dot_prod_f32(a, b, 4, &result);
    return result;
}

void test_cmsis_nn(UART_HandleTypeDef *huart)
{
    char buf[128];

    // Mix of positive/negative q7 values (-128 to 127)
    q7_t data[] = { -10, -5, 0, 5, 10, -128, 127, -1, 1, -50 };
    const uint16_t len = sizeof(data) / sizeof(data[0]);

    // Print input
    snprintf(buf, sizeof(buf), "Input:  ");
    HAL_UART_Transmit(huart, (uint8_t*)buf, strlen(buf), HAL_MAX_DELAY);
    for (int i = 0; i < len; i++) {
        snprintf(buf, sizeof(buf), "%5d", (int)data[i]);
        HAL_UART_Transmit(huart, (uint8_t*)buf, strlen(buf), HAL_MAX_DELAY);
    }
    HAL_UART_Transmit(huart, (uint8_t*)"\r\n", 2, HAL_MAX_DELAY);

    // Run CMSIS-NN ReLU
    arm_relu_q7(data, len);

    // Print output
    snprintf(buf, sizeof(buf), "ReLU:   ");
    HAL_UART_Transmit(huart, (uint8_t*)buf, strlen(buf), HAL_MAX_DELAY);
    for (int i = 0; i < len; i++) {
        snprintf(buf, sizeof(buf), "%5d", (int)data[i]);
        HAL_UART_Transmit(huart, (uint8_t*)buf, strlen(buf), HAL_MAX_DELAY);
    }
    HAL_UART_Transmit(huart, (uint8_t*)"\r\n", 2, HAL_MAX_DELAY);
}




// Taken from SNNBoardBraille7_Test folder in the thesis (main.c)
void example_usage(void) {
    
  usart1_print("THE SIMULATION BEGINS...\r\n");
  usart1_print(" \r\n");
  SNN_Init(); // Initialize the Spiking Neural Network

  q7_t input_spikes[12];
  q7_t output_spikes[7]; // Output spikes for 7 output neurons
  // No float conversion needed; outputs are binary spikes (0/1)
  uint32_t num_matches = 0;

  for (int i = 0; i < TEST_OUTPUT; i++) {

  // Reset accumulate output at each new input
  memset(accumulate_output, 0, sizeof(accumulate_output));

    for (int j = 0; j < NUM_STEPS; j++) {
      // Set input spikes for this timestep (fill 12 inputs for 256 timesteps)
      for (int k = 0; k < INPUTS; k++) {
        input_spikes[k] = test_input[i][j][k]; //take each 12 elements array to compute, 256 for each classification until 140 classifications
      }
      SNN_Run_Timestep(input_spikes, output_spikes);
      // accumulate integer spike counts for classification
      for (int o = 0; o < OUTPUTS; o++) {
        /* output_spikes[o] is q7_t with values 0 or 1; add to int32 counter */
        accumulate_output[o] += output_spikes[o];
      }
    }
    //Now print the accumulate output, the class predicted and the class expected
    usart1_print("Output spikes accumulation: ");
    for (int n = 0; n < OUTPUTS; n++) {
      char msg[32];
      /* accumulate_output is int32_t[] - print as integer counts */
      snprintf(msg, sizeof(msg), "%lu ", accumulate_output[n]);
      usart1_print(msg);
    }
    usart1_print(" \r\n");
    //predicted class
  uint8_t predicted_class = 0;
  /* find index of max count in integer accumulation */
  int32_t max_val = accumulate_output[0];
  for (uint8_t idx = 1; idx < OUTPUTS; idx++) {
    if (accumulate_output[idx] > max_val) {
      max_val = accumulate_output[idx];
      predicted_class = idx;
    }
  }
    char msg_pred[50];
    snprintf(msg_pred, sizeof(msg_pred), "Predicted class: %u\r\n", predicted_class);
    usart1_print(msg_pred);
    //expected class
  char msg_exp[50];
  /* assume class_output is an integer type small enough for %u */
  snprintf(msg_exp, sizeof(msg_exp), "Expected class: %u\r\n", (unsigned)class_output[i]);
    usart1_print(msg_exp);
    usart1_print(" \r\n");
    
    if (predicted_class == class_output[i]) {
      usart1_print("Match\r\n");
      num_matches++;
    } else {
      usart1_print("Miss\r\n");
    }
  }

  char msg_final[80];
  float accuracy = ((float)num_matches / (float)TEST_OUTPUT) * 100.0f;
  /* print num_matches and TEST_OUTPUT as unsigned integers */
  snprintf(msg_final, sizeof(msg_final), "Simulation completed. Accuracy: %.2f%% (%u/%u)\r\n", accuracy, (unsigned)num_matches, (unsigned)TEST_OUTPUT);
  usart1_print(msg_final);

}



  char msg[] = "U are sane, board works!\r\n";
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    HAL_UART_Transmit(&huart3, (uint8_t*)msg, strlen(msg), 100);
    HAL_GPIO_TogglePin(GPIOI, GPIO_PIN_13);  
    HAL_Delay(1000);
    HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_3);

    // char buf[32];
    // result = dsp_test();
    // int len = snprintf(buf, sizeof(buf), "Result= %d\r\n", (int) result);
    // HAL_UART_Transmit(&huart3, (uint8_t*)buf, len, 100);


    //test_cmsis_nn(&huart3); 

    example_usage();
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI|RCC_OSCILLATORTYPE_LSI
                              |RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.LSIState = RCC_LSI_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 2;
  RCC_OscInitStruct.PLL.PLLN = 12;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 3;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOMEDIUM;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief Peripherals Common Clock Configuration
  * @retval None
  */
void PeriphCommonClock_Config(void)
{
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Initializes the peripherals clock
  */
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_ADC;
  PeriphClkInitStruct.PLL2.PLL2M = 2;
  PeriphClkInitStruct.PLL2.PLL2N = 12;
  PeriphClkInitStruct.PLL2.PLL2P = 2;
  PeriphClkInitStruct.PLL2.PLL2Q = 2;
  PeriphClkInitStruct.PLL2.PLL2R = 2;
  PeriphClkInitStruct.PLL2.PLL2RGE = RCC_PLL2VCIRANGE_3;
  PeriphClkInitStruct.PLL2.PLL2VCOSEL = RCC_PLL2VCOMEDIUM;
  PeriphClkInitStruct.PLL2.PLL2FRACN = 0;
  PeriphClkInitStruct.AdcClockSelection = RCC_ADCCLKSOURCE_PLL2;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
