export type ParameterType = 'temperature'| 'heatIndex' | 'humidity' | 'pressure' | 'windSpeed' | 'precipitation' | 'uvIndex' | 'lightIntensity';

export interface ParameterDataPoint {
  id: number;
  recordedAt: string;
  value: number;
}

export interface ParameterConfig {
  key: ParameterType;
  apiKey: string; 
  label: string;
  color: string;
  unit: string;
  chartType?: 'line' | 'area' | 'bar';
}
