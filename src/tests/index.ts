import { kernelTests, type SelfTest } from './kernels';
import { modelTests } from './model';

export type { SelfTest, TestResult } from './kernels';

export const allTests: SelfTest[] = [...kernelTests, ...modelTests];