//! Process exit codes from `console/nii_dicom.h`.

use std::process::ExitCode;

/// dcm2niix exit statuses. Scripts branch on these; keep the discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Exit {
    Success = 0,
    Failure = 1,
    NoValidFilesFound = 2,
    ReportVersion = 3,
    CorruptFileFound = 4,
    InputFolderInvalid = 5,
    OutputFolderInvalid = 6,
    OutputFolderReadOnly = 7,
    SomeOkSomeBad = 8,
    RenameError = 9,
    IncompleteVolumesFound = 10,
    Nominal = 11,
    InvalidParam = 12,
}

impl Exit {
    pub fn code(self) -> u8 {
        self as u8
    }

    pub fn to_exit_code(self) -> ExitCode {
        ExitCode::from(self.code())
    }
}
