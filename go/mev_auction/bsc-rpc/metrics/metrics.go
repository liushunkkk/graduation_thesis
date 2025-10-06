// Go port of Coda Hale's Metrics library
//
// <https://github.com/rcrowley/go-metrics>
//
// Coda Hale's original work: <https://github.com/codahale/metrics>
package metrics

import (
	"fmt"
	"github.com/shirou/gopsutil/v4/disk"
	"github.com/shirou/gopsutil/v4/mem"
	"github.com/shirou/gopsutil/v4/process"
	"os"
	"os/exec"
	"runtime/metrics"
	"runtime/pprof"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/ethereum/go-ethereum/log"
)

// Enabled is checked by the constructor functions for all of the
// standard metrics. If it is true, the metric returned is a stub.
//
// This global kill-switch helps quantify the observer effect and makes
// for less cluttered pprof profiles.
var Enabled = false

// EnabledExpensive is a soft-flag meant for external packages to check if costly
// metrics gathering is allowed or not. The goal is to separate standard metrics
// for health monitoring and debug metrics that might impact runtime performance.
var EnabledExpensive = false

// enablerFlags is the CLI flag names to use to enable metrics collections.
var enablerFlags = []string{"metrics"}

// enablerEnvVars is the env var names to use to enable metrics collections.
var enablerEnvVars = []string{"GETH_METRICS"}

// expensiveEnablerFlags is the CLI flag names to use to enable metrics collections.
var expensiveEnablerFlags = []string{"metrics.expensive"}

// expensiveEnablerEnvVars is the env var names to use to enable metrics collections.
var expensiveEnablerEnvVars = []string{"GETH_METRICS_EXPENSIVE"}

// Init enables or disables the metrics system. Since we need this to run before
// any other code gets to create meters and timers, we'll actually do an ugly hack
// and peek into the command line args for the metrics flag.
func init() {
	for _, enabler := range enablerEnvVars {
		if val, found := syscall.Getenv(enabler); found && !Enabled {
			if enable, _ := strconv.ParseBool(val); enable { // ignore error, flag parser will choke on it later
				log.Info("Enabling metrics collection")
				Enabled = true
			}
		}
	}
	for _, enabler := range expensiveEnablerEnvVars {
		if val, found := syscall.Getenv(enabler); found && !EnabledExpensive {
			if enable, _ := strconv.ParseBool(val); enable { // ignore error, flag parser will choke on it later
				log.Info("Enabling expensive metrics collection")
				EnabledExpensive = true
			}
		}
	}
	for _, arg := range os.Args {
		flag := strings.TrimLeft(arg, "-")

		for _, enabler := range enablerFlags {
			if !Enabled && flag == enabler {
				log.Info("Enabling metrics collection")
				Enabled = true
			}
		}
		for _, enabler := range expensiveEnablerFlags {
			if !EnabledExpensive && flag == enabler {
				log.Info("Enabling expensive metrics collection")
				EnabledExpensive = true
			}
		}
	}
}

var threadCreateProfile = pprof.Lookup("threadcreate")

type runtimeStats struct {
	GCPauses     *metrics.Float64Histogram
	GCAllocBytes uint64
	GCFreedBytes uint64

	MemTotal     uint64
	HeapObjects  uint64
	HeapFree     uint64
	HeapReleased uint64
	HeapUnused   uint64

	Goroutines   uint64
	SchedLatency *metrics.Float64Histogram
}

var runtimeSamples = []metrics.Sample{
	{Name: "/gc/pauses:seconds"}, // histogram
	{Name: "/gc/heap/allocs:bytes"},
	{Name: "/gc/heap/frees:bytes"},
	{Name: "/memory/classes/total:bytes"},
	{Name: "/memory/classes/heap/objects:bytes"},
	{Name: "/memory/classes/heap/free:bytes"},
	{Name: "/memory/classes/heap/released:bytes"},
	{Name: "/memory/classes/heap/unused:bytes"},
	{Name: "/sched/goroutines:goroutines"},
	{Name: "/sched/latencies:seconds"}, // histogram
}

func ReadRuntimeStats() *runtimeStats {
	r := new(runtimeStats)
	readRuntimeStats(r)
	return r
}

func readRuntimeStats(v *runtimeStats) {
	metrics.Read(runtimeSamples)
	for _, s := range runtimeSamples {
		// Skip invalid/unknown metrics. This is needed because some metrics
		// are unavailable in older Go versions, and attempting to read a 'bad'
		// metric panics.
		if s.Value.Kind() == metrics.KindBad {
			continue
		}

		switch s.Name {
		case "/gc/pauses:seconds":
			v.GCPauses = s.Value.Float64Histogram()
		case "/gc/heap/allocs:bytes":
			v.GCAllocBytes = s.Value.Uint64()
		case "/gc/heap/frees:bytes":
			v.GCFreedBytes = s.Value.Uint64()
		case "/memory/classes/total:bytes":
			v.MemTotal = s.Value.Uint64()
		case "/memory/classes/heap/objects:bytes":
			v.HeapObjects = s.Value.Uint64()
		case "/memory/classes/heap/free:bytes":
			v.HeapFree = s.Value.Uint64()
		case "/memory/classes/heap/released:bytes":
			v.HeapReleased = s.Value.Uint64()
		case "/memory/classes/heap/unused:bytes":
			v.HeapUnused = s.Value.Uint64()
		case "/sched/goroutines:goroutines":
			v.Goroutines = s.Value.Uint64()
		case "/sched/latencies:seconds":
			v.SchedLatency = s.Value.Float64Histogram()
		}
	}
}

// CollectProcessMetrics periodically collects various metrics about the running process.
func CollectProcessMetrics(refresh time.Duration) {
	// Short circuit if the metrics system is disabled
	if !Enabled {
		return
	}

	go CollectSystemMetrics(refresh)

	// Create the various data collectors
	var (
		cpustats  = make([]CPUStats, 2)
		diskstats = make([]DiskStats, 2)
		rstats    = make([]runtimeStats, 2)
	)

	// This scale factor is used for the runtime's time metrics. It's useful to convert to
	// ns here because the runtime gives times in float seconds, but runtimeHistogram can
	// only provide integers for the minimum and maximum values.
	const secondsToNs = float64(time.Second)

	// Define the various metrics to collect
	var (
		cpuSysLoad              = GetOrRegisterGauge("system/cpu/sysload", DefaultRegistry)
		cpuSysWait              = GetOrRegisterGauge("system/cpu/syswait", DefaultRegistry)
		cpuProcLoad             = GetOrRegisterGauge("system/cpu/procload", DefaultRegistry)
		cpuSysLoadTotal         = GetOrRegisterCounterFloat64("system/cpu/sysload/total", DefaultRegistry)
		cpuSysWaitTotal         = GetOrRegisterCounterFloat64("system/cpu/syswait/total", DefaultRegistry)
		cpuProcLoadTotal        = GetOrRegisterCounterFloat64("system/cpu/procload/total", DefaultRegistry)
		cpuThreads              = GetOrRegisterGauge("system/cpu/threads", DefaultRegistry)
		cpuGoroutines           = GetOrRegisterGauge("system/cpu/goroutines", DefaultRegistry)
		cpuSchedLatency         = getOrRegisterRuntimeHistogram("system/cpu/schedlatency", secondsToNs, nil)
		memPauses               = getOrRegisterRuntimeHistogram("system/memory/pauses", secondsToNs, nil)
		memAllocs               = GetOrRegisterMeter("system/memory/allocs", DefaultRegistry)
		memFrees                = GetOrRegisterMeter("system/memory/frees", DefaultRegistry)
		memTotal                = GetOrRegisterGauge("system/memory/held", DefaultRegistry)
		heapUsed                = GetOrRegisterGauge("system/memory/used", DefaultRegistry)
		heapObjects             = GetOrRegisterGauge("system/memory/objects", DefaultRegistry)
		diskReads               = GetOrRegisterMeter("system/disk/readcount", DefaultRegistry)
		diskReadBytes           = GetOrRegisterMeter("system/disk/readdata", DefaultRegistry)
		diskReadBytesCounter    = GetOrRegisterCounter("system/disk/readbytes", DefaultRegistry)
		diskWrites              = GetOrRegisterMeter("system/disk/writecount", DefaultRegistry)
		diskWriteBytes          = GetOrRegisterMeter("system/disk/writedata", DefaultRegistry)
		diskWriteBytesCounter   = GetOrRegisterCounter("system/disk/writebytes", DefaultRegistry)
		diskIOReadBytesCounter  = GetOrRegisterCounter("system/disk/io/readbytes", DefaultRegistry)
		diskIOWriteBytesCounter = GetOrRegisterCounter("system/disk/io/writebytes", DefaultRegistry)
	)

	var lastCollectTime time.Time

	// Iterate loading the different stats and updating the meters.
	now, prev := 0, 1
	for ; ; now, prev = prev, now {
		// Gather CPU times.
		ReadCPUStats(&cpustats[now])
		collectTime := time.Now()
		secondsSinceLastCollect := collectTime.Sub(lastCollectTime).Seconds()
		lastCollectTime = collectTime
		if secondsSinceLastCollect > 0 {
			sysLoad := cpustats[now].GlobalTime - cpustats[prev].GlobalTime
			sysWait := cpustats[now].GlobalWait - cpustats[prev].GlobalWait
			procLoad := cpustats[now].LocalTime - cpustats[prev].LocalTime
			// Convert to integer percentage.
			cpuSysLoad.Update(int64(sysLoad / secondsSinceLastCollect * 100))
			cpuSysWait.Update(int64(sysWait / secondsSinceLastCollect * 100))
			cpuProcLoad.Update(int64(procLoad / secondsSinceLastCollect * 100))
			// increment counters (ms)
			cpuSysLoadTotal.Inc(sysLoad)
			cpuSysWaitTotal.Inc(sysWait)
			cpuProcLoadTotal.Inc(procLoad)
		}

		// Threads
		cpuThreads.Update(int64(threadCreateProfile.Count()))

		// Go runtime metrics
		readRuntimeStats(&rstats[now])

		cpuGoroutines.Update(int64(rstats[now].Goroutines))
		cpuSchedLatency.update(rstats[now].SchedLatency)
		memPauses.update(rstats[now].GCPauses)

		memAllocs.Mark(int64(rstats[now].GCAllocBytes - rstats[prev].GCAllocBytes))
		memFrees.Mark(int64(rstats[now].GCFreedBytes - rstats[prev].GCFreedBytes))

		memTotal.Update(int64(rstats[now].MemTotal))
		heapUsed.Update(int64(rstats[now].MemTotal - rstats[now].HeapUnused - rstats[now].HeapFree - rstats[now].HeapReleased))
		heapObjects.Update(int64(rstats[now].HeapObjects))

		// Disk
		if ReadDiskStats(&diskstats[now]) == nil {
			diskReads.Mark(diskstats[now].ReadCount - diskstats[prev].ReadCount)
			diskReadBytes.Mark(diskstats[now].ReadBytes - diskstats[prev].ReadBytes)
			diskWrites.Mark(diskstats[now].WriteCount - diskstats[prev].WriteCount)
			diskWriteBytes.Mark(diskstats[now].WriteBytes - diskstats[prev].WriteBytes)
			diskReadBytesCounter.Inc(diskstats[now].ReadBytes - diskstats[prev].ReadBytes)
			diskWriteBytesCounter.Inc(diskstats[now].WriteBytes - diskstats[prev].WriteBytes)
			diskIOReadBytesCounter.Inc(diskstats[now].ReadIOBytes - diskstats[prev].ReadIOBytes)
			diskIOWriteBytesCounter.Inc(diskstats[now].WriteIOBytes - diskstats[prev].WriteIOBytes)
		}

		time.Sleep(refresh)
	}
}

// CollectSystemMetrics 使用 gopsutil 收集系统信息：cpu, disk and memory.
func CollectSystemMetrics(refresh time.Duration) {
	// Create the various data collectors
	var (
		cpustats  = make([]CPUStats, 2)
		diskstats = make([]ScutumDiskStats, 2)
		rstats    = make([]runtimeStats, 2)
	)

	// Define the various metrics to collect
	var (
		cpuSysLoad       = GetOrRegisterGauge("scutum/system/cpu/sysload", DefaultRegistry)
		cpuSysWait       = GetOrRegisterGauge("scutum/system/cpu/syswait", DefaultRegistry)
		cpuProcLoad      = GetOrRegisterGauge("scutum/system/cpu/procload", DefaultRegistry)
		cpuSysLoadTotal  = GetOrRegisterCounterFloat64("scutum/system/cpu/sysload/total", DefaultRegistry)
		cpuSysWaitTotal  = GetOrRegisterCounterFloat64("scutum/system/cpu/syswait/total", DefaultRegistry)
		cpuProcLoadTotal = GetOrRegisterCounterFloat64("scutum/system/cpu/procload/total", DefaultRegistry)
		cpuThreads       = GetOrRegisterGauge("scutum/system/cpu/threads", DefaultRegistry)
		cpuGoroutines    = GetOrRegisterGauge("scutum/system/cpu/goroutines", DefaultRegistry)

		memTotal     = GetOrRegisterGauge("scutum/system/memory/total", DefaultRegistry)
		memFree      = GetOrRegisterGauge("scutum/system/memory/free", DefaultRegistry)
		memAvailable = GetOrRegisterGauge("scutum/system/memory/available", DefaultRegistry)
		memUsed      = GetOrRegisterGauge("scutum/system/memory/used", DefaultRegistry)
		memGethUsed  = GetOrRegisterGauge("scutum/system/memory/gethused", DefaultRegistry)

		diskReadCounter  = GetOrRegisterCounter("scutum/system/disk/readcount", DefaultRegistry)
		diskWriteCounter = GetOrRegisterCounter("scutum/system/disk/writecount", DefaultRegistry)
		readBytes        = GetOrRegisterCounter("scutum/system/disk/readbytes", DefaultRegistry)
		writeBytes       = GetOrRegisterCounter("scutum/system/disk/writebytes", DefaultRegistry)
	)

	// 使用 pgrep 查找进程名为 geth-linux 的进程 PID
	cmd := exec.Command("pgrep", "geth-linux")
	output, err := cmd.Output()
	if err != nil {
		fmt.Println("未找到进程或发生错误:", err)
		return
	}
	// 去除输出中的换行符
	pidStr := strings.TrimSpace(string(output))
	pid, err := strconv.Atoi(pidStr)
	if err != nil {
		fmt.Printf("geth-linux 进程 pid 解析失败")
		return
	}
	fmt.Printf("geth-linux 进程的 PID 是: %v\n", pid)

	var mountDirs []string
	gethProcess, err := process.NewProcess(int32(pid))
	if err != nil {
		gethProcess = nil
	}
	// 获取进程启动命令行
	if gethProcess != nil {
		workingDir, err := gethProcess.Cwd()
		if err == nil {
			mountDirs = extractMountDir(workingDir)
		}
	}

	var lastCollectTime time.Time

	// Iterate loading the different stats and updating the meters.
	now, prev := 0, 1
	for ; ; now, prev = prev, now {
		// Gather CPU times.
		ReadScutumCPUStats(&cpustats[now])
		collectTime := time.Now()
		secondsSinceLastCollect := collectTime.Sub(lastCollectTime).Seconds()
		lastCollectTime = collectTime
		if secondsSinceLastCollect > 0 {
			sysLoad := cpustats[now].GlobalTime - cpustats[prev].GlobalTime
			sysWait := cpustats[now].GlobalWait - cpustats[prev].GlobalWait
			procLoad := cpustats[now].LocalTime - cpustats[prev].LocalTime
			// Convert to integer percentage.
			cpuSysLoad.Update(int64(sysLoad / secondsSinceLastCollect * 100))
			cpuSysWait.Update(int64(sysWait / secondsSinceLastCollect * 100))
			cpuProcLoad.Update(int64(procLoad / secondsSinceLastCollect * 100))
			// increment counters (ms)
			cpuSysLoadTotal.Inc(sysLoad)
			cpuSysWaitTotal.Inc(sysWait)
			cpuProcLoadTotal.Inc(procLoad)
		}

		// Threads
		if gethProcess != nil {
			threads, err := gethProcess.NumThreads()
			if err == nil {
				cpuThreads.Update(int64(threads))
			}
			memoryInfo, err := gethProcess.MemoryInfo()
			if err == nil {
				memGethUsed.Update(int64(memoryInfo.RSS))
			}
		}

		// Go runtime metrics
		readScutumRuntimeStats(&rstats[now])
		cpuGoroutines.Update(int64(rstats[now].Goroutines))

		virtualMemory, err := mem.VirtualMemory()
		if err == nil {
			memAvailable.Update(int64(virtualMemory.Available))
			memFree.Update(int64(virtualMemory.Free))
			memUsed.Update(int64(virtualMemory.Used))
			memTotal.Update(int64(virtualMemory.Total))
		}

		// Disk
		collectDiskMetrics(mountDirs)
		if ReadScutumDiskStats(&diskstats[now]) == nil {
			diskReadCounter.Inc(diskstats[now].ReadCount - diskstats[prev].ReadCount)
			diskWriteCounter.Inc(diskstats[now].WriteCount - diskstats[prev].WriteCount)
			readBytes.Inc(diskstats[now].ReadBytes - diskstats[prev].ReadBytes)
			writeBytes.Inc(diskstats[now].WriteBytes - diskstats[prev].WriteBytes)
		}

		time.Sleep(refresh)
	}
}

func collectDiskMetrics(mounts []string) {
	for _, mount := range mounts {
		k := ""
		if mount == "/" {
			k = "/bscdev"
		} else if mount == "/mnt" {
			k = "/ethdev"
		} else if strings.HasPrefix(mount, "/mnt") {
			k = mount[4:]
		}
		var (
			diskTotal = GetOrRegisterGauge("scutum/system/disk"+k+"/total", DefaultRegistry)
			diskUsed  = GetOrRegisterGauge("scutum/system/disk"+k+"/used", DefaultRegistry)
			diskFree  = GetOrRegisterGauge("scutum/system/disk"+k+"/free", DefaultRegistry)
		)
		// Disk
		diskUsage, err := disk.Usage(mount)
		if err == nil {
			diskTotal.Update(int64(diskUsage.Total))
			diskUsed.Update(int64(diskUsage.Used))
			diskFree.Update(int64(diskUsage.Free))
		}
	}
}

func extractMountDir(workingDir string) []string {
	partitions, err := disk.Partitions(true)
	if err != nil {
		return []string{}
	}
	var mounts []string
	for _, partition := range partitions {
		if strings.Contains("/root/geth.fast", workingDir) {
			// bsc dev
			if partition.Mountpoint == "/" {
				mounts = append(mounts, partition.Mountpoint)
			}
		} else if strings.Contains("/mnt/data", workingDir) {
			// eth dev
			if partition.Mountpoint == "/mnt" {
				mounts = append(mounts, partition.Mountpoint)
			}
		} else {
			if strings.HasPrefix(partition.Mountpoint, "/mnt") {
				mounts = append(mounts, partition.Mountpoint)
			}
		}
	}
	return mounts
}

func readScutumRuntimeStats(v *runtimeStats) {
	metrics.Read(runtimeSamples)
	for _, s := range runtimeSamples {
		// Skip invalid/unknown metrics. This is needed because some metrics
		// are unavailable in older Go versions, and attempting to read a 'bad'
		// metric panics.
		if s.Value.Kind() == metrics.KindBad {
			continue
		}

		switch s.Name {
		case "/sched/goroutines:goroutines":
			v.Goroutines = s.Value.Uint64()
		}
	}
}

type ScutumDiskStats struct {
	ReadCount  int64 // Number of read operations executed
	ReadBytes  int64 // Total number of bytes read (include disk cache)
	WriteCount int64 // Number of write operations executed
	WriteBytes int64 // Total number of byte written
}

// ReadScutumDiskStats retrieves the disk IO stats for whole system disk.
func ReadScutumDiskStats(stats *ScutumDiskStats) error {
	counters, err := disk.IOCounters()
	if err != nil {
		fmt.Printf("获取 disk IOCounters 失败")
		return err
	}
	var totalReadCount, totalWriteCount, totalReadBytes, totalWriteBytes uint64
	for _, counter := range counters {
		totalReadCount += counter.ReadCount
		totalReadBytes += counter.ReadBytes
		totalWriteCount += counter.WriteCount
		totalWriteBytes += counter.WriteBytes
	}
	stats.ReadCount = int64(totalReadCount)
	stats.ReadBytes = int64(totalReadBytes)
	stats.WriteCount = int64(totalWriteCount)
	stats.WriteBytes = int64(totalWriteBytes)
	return nil
}
